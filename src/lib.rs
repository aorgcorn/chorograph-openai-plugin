use chorograph_plugin_sdk_rust::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OpenAI-compatible REST API plugin (WASM path) — streaming + agentic tool calls
// ---------------------------------------------------------------------------
// Reads configuration from the host UserDefaults via `get_user_default`:
//   "openaiCompatibleBaseURL"  — base URL, e.g. "http://localhost:11434"
//   "openaiCompatibleAPIKey"   — API key (may be empty for local models)
//   "openaiCompatibleModel"    — model ID, e.g. "gpt-4o" or "llama3"
// ---------------------------------------------------------------------------

struct OpenAIPlugin;

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// Flexible OpenAI message.  For user/assistant text turns only `role` and
/// `content` are set.  For an assistant turn that requested tool calls,
/// `content` may be null and `tool_calls` is populated.  For a tool-result
/// message, `role` is "tool", `content` holds the result, and
/// `tool_call_id` links back to the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Present when role == "assistant" and the model requested tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<AssistantToolCall>>,
    /// Present when role == "tool".
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAIMessage {
    fn assistant_tool_calls(calls: Vec<AssistantToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }
    fn tool_result(call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(result.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// A tool call as it appears inside an assistant message sent back to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: AssistantToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AssistantToolCallFunction {
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// Streaming response types (for `"stream": true`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct ToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAIStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAIStreamChoice>,
}

// ---------------------------------------------------------------------------
// Non-streaming response types (used by fetch_models)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAIModel {
    id: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

// ---------------------------------------------------------------------------
// Agentic loop result
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum FinishReason {
    Stop,
    ToolCalls,
    Unknown,
}

/// A fully-accumulated tool call from a streaming response.
#[derive(Debug, Clone)]
struct CompletedToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Result of one streaming round.
struct StreamResult {
    finish_reason: FinishReason,
    tool_calls: Vec<CompletedToolCall>,
}

// ---------------------------------------------------------------------------
// Tool definitions sent to the API
// ---------------------------------------------------------------------------

fn tool_definitions() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a file from the workspace. Use this to inspect source code, configuration files, or any text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or workspace-relative file path to read."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the workspace root directory and return its combined stdout and stderr output. Use for searching, listing files, running tests, building, or any shell operation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute (runs via sh -c in the workspace root)."
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ])
}

// ---------------------------------------------------------------------------
// Plugin implementation
// ---------------------------------------------------------------------------

impl OpenAIPlugin {
    fn require_default(key: &str) -> Result<String> {
        get_user_default(key).ok_or_else(|| {
            PluginError::Other(format!(
                "OpenAI plugin: '{}' is not configured. Please set it in Settings.",
                key
            ))
        })
    }

    fn auth_header(api_key: &str) -> Option<String> {
        if api_key.is_empty() {
            None
        } else {
            Some(format!("Bearer {}", api_key))
        }
    }

    // -----------------------------------------------------------------------
    // Tool executor
    // -----------------------------------------------------------------------

    /// Execute a tool call and return the result string.
    fn execute_tool(name: &str, arguments_json: &str) -> String {
        match name {
            "read_file" => Self::tool_read_file(arguments_json),
            "bash" => Self::tool_bash(arguments_json),
            other => format!("Unknown tool: {}", other),
        }
    }

    fn tool_read_file(arguments_json: &str) -> String {
        let args: serde_json::Value = match serde_json::from_str(arguments_json) {
            Ok(v) => v,
            Err(e) => return format!("Failed to parse arguments: {}", e),
        };
        let path = match args.get("path").and_then(|p| p.as_str()) {
            Some(p) => p.to_string(),
            None => return "Missing required argument: path".into(),
        };
        push_tool_call(&format!("READ {}", path));
        match read_host_file(&path) {
            Ok(content) => content,
            Err(e) => format!("Error reading file '{}': {:?}", path, e),
        }
    }

    fn tool_bash(arguments_json: &str) -> String {
        let args: serde_json::Value = match serde_json::from_str(arguments_json) {
            Ok(v) => v,
            Err(e) => return format!("Failed to parse arguments: {}", e),
        };
        let command = match args.get("command").and_then(|c| c.as_str()) {
            Some(c) => c.to_string(),
            None => return "Missing required argument: command".into(),
        };
        push_tool_call(&format!("BASH {}", command));

        let cwd = get_user_default("serverDirectory").unwrap_or_default();
        let cwd_opt: Option<&str> = if cwd.is_empty() {
            None
        } else {
            Some(cwd.as_str())
        };

        let proc = match ChildProcess::spawn("sh", vec!["-c", &command], cwd_opt, HashMap::new()) {
            Ok(p) => p,
            Err(e) => return format!("Failed to spawn process: {:?}", e),
        };

        // Drain stdout and stderr until EOF, with a reasonable timeout between reads.
        let mut output = String::new();
        let max_bytes = 32 * 1024; // cap at 32 KB to avoid overwhelming context
        let mut stdout_done = false;
        let mut stderr_done = false;

        loop {
            if output.len() >= max_bytes {
                output.push_str("\n[output truncated at 32 KB]");
                break;
            }

            if !stdout_done {
                match proc.read(PipeType::Stdout) {
                    Ok(ReadResult::Data(bytes)) => {
                        if let Ok(s) = String::from_utf8(bytes) {
                            output.push_str(&s);
                        }
                        continue;
                    }
                    Ok(ReadResult::EOF) => stdout_done = true,
                    Ok(ReadResult::Empty) => {}
                    Err(_) => stdout_done = true,
                }
            }

            if !stderr_done {
                match proc.read(PipeType::Stderr) {
                    Ok(ReadResult::Data(bytes)) => {
                        if let Ok(s) = String::from_utf8(bytes) {
                            output.push_str(&s);
                        }
                        continue;
                    }
                    Ok(ReadResult::EOF) => stderr_done = true,
                    Ok(ReadResult::Empty) => {}
                    Err(_) => stderr_done = true,
                }
            }

            if stdout_done && stderr_done {
                break;
            }

            // Wait up to 5 s for more data.
            if !proc.wait_for_data(5_000) {
                break; // timeout
            }
        }

        if output.is_empty() {
            "(no output)".into()
        } else {
            output
        }
    }

    // -----------------------------------------------------------------------
    // Streaming round
    // -----------------------------------------------------------------------

    /// Stream one chat completion round.  Returns the finish reason and any
    /// accumulated tool calls.  Emits `StreamingDelta` events for content tokens.
    fn stream_one_round(
        session_id: &str,
        base_url: &str,
        api_key: &str,
        model: &str,
        messages: &[OpenAIMessage],
    ) -> StreamResult {
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

        let body = match serde_json::to_string(&json!({
            "model": model,
            "messages": messages,
            "tools": tool_definitions(),
            "tool_choice": "auto",
            "stream": true
        })) {
            Ok(b) => b,
            Err(e) => {
                push_ai_event(
                    session_id,
                    &AIEvent::Error {
                        message: format!("Failed to serialise request: {}", e),
                    },
                );
                return StreamResult {
                    finish_reason: FinishReason::Unknown,
                    tool_calls: vec![],
                };
            }
        };

        let mut headers: Vec<(&str, String)> = vec![
            ("Content-Type", "application/json".to_string()),
            ("Accept", "text/event-stream".to_string()),
        ];
        let auth_value;
        if let Some(auth) = Self::auth_header(api_key) {
            auth_value = auth;
            headers.push(("Authorization", auth_value.clone()));
        }
        let header_refs: Vec<(&str, &str)> =
            headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

        push_tool_call(&format!(
            "CALL {} → {}",
            model,
            base_url.trim_end_matches('/')
        ));

        let handle = match sse_post(&url, Some(&header_refs), &body) {
            Ok(h) => h,
            Err(e) => {
                push_ai_event(
                    session_id,
                    &AIEvent::Error {
                        message: format!("SSE connection failed: {}", e),
                    },
                );
                return StreamResult {
                    finish_reason: FinishReason::Unknown,
                    tool_calls: vec![],
                };
            }
        };

        // Shared state accumulated across SSE chunks — use a Cell-wrapped struct
        // because for_each_sse_line takes FnMut (not FnOnce).
        let sid = session_id.to_string();
        let mut finish_reason = FinishReason::Unknown;
        // tool_calls_acc: index → (id, name, arguments_so_far)
        let mut tool_calls_acc: Vec<(String, String, String)> = Vec::new();

        for_each_sse_line(handle, |line| {
            let data = match line.strip_prefix("data: ") {
                Some(d) => d.trim(),
                None => return true,
            };

            if data == "[DONE]" {
                return false;
            }
            if data.is_empty() {
                return true;
            }

            let chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => {
                    log!("[OpenAI Plugin] SSE parse error: {} — {:?}", data, e);
                    return true;
                }
            };

            for choice in chunk.choices {
                // Capture finish_reason whenever it appears.
                if let Some(ref fr) = choice.finish_reason {
                    match fr.as_str() {
                        "stop" => finish_reason = FinishReason::Stop,
                        "tool_calls" => finish_reason = FinishReason::ToolCalls,
                        _ => {}
                    }
                }

                let delta = choice.delta;

                // Content token → emit StreamingDelta.
                if let Some(text) = delta.content {
                    if !text.is_empty() {
                        push_ai_event(
                            &sid,
                            &AIEvent::StreamingDelta {
                                session_id: sid.clone(),
                                text,
                            },
                        );
                    }
                }

                // Tool call deltas → accumulate into tool_calls_acc.
                if let Some(tc_deltas) = delta.tool_calls {
                    for tc in tc_deltas {
                        let idx = tc.index;
                        // Grow the accumulator if needed.
                        while tool_calls_acc.len() <= idx {
                            tool_calls_acc.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(func) = tc.function {
                            if let Some(name) = func.name {
                                tool_calls_acc[idx].1.push_str(&name);
                            }
                            if let Some(args) = func.arguments {
                                tool_calls_acc[idx].2.push_str(&args);
                            }
                        }
                        if let Some(id) = tc.id {
                            tool_calls_acc[idx].0 = id;
                        }
                    }
                }
            }

            true
        });

        log!(
            "[OpenAI Plugin] SSE round done finish_reason={:?} tool_calls={}",
            finish_reason,
            tool_calls_acc.len()
        );

        let completed: Vec<CompletedToolCall> = tool_calls_acc
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, arguments)| CompletedToolCall {
                id,
                name,
                arguments,
            })
            .collect();

        StreamResult {
            finish_reason,
            tool_calls: completed,
        }
    }

    // -----------------------------------------------------------------------
    // Models fetch
    // -----------------------------------------------------------------------

    fn fetch_models(base_url: &str, api_key: &str) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
        let mut headers: Vec<(&str, String)> = vec![("Accept", "application/json".to_string())];
        let auth_value;
        if let Some(auth) = Self::auth_header(api_key) {
            auth_value = auth;
            headers.push(("Authorization", auth_value.clone()));
        }
        let header_refs: Vec<(&str, &str)> =
            headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let resp = http_get(&url, Some(&header_refs))?;
        if resp.status < 200 || resp.status >= 300 {
            return Err(PluginError::Other(format!(
                "GET /v1/models returned HTTP {}: {}",
                resp.status, resp.body
            )));
        }
        let parsed: OpenAIModelsResponse = serde_json::from_str(&resp.body).map_err(|e| {
            PluginError::SerializationError(format!("Failed to parse models: {}", e))
        })?;
        Ok(parsed.data.into_iter().map(|m| m.id).collect())
    }

    // -----------------------------------------------------------------------
    // Message conversion (Chorograph history → OpenAI)
    // -----------------------------------------------------------------------

    fn to_openai_messages(messages: &[serde_json::Value]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .filter_map(|m| {
                let role = m.get("role").and_then(|r| r.as_str())?;
                let text = m.get("text").and_then(|t| t.as_str()).unwrap_or("");
                Some(OpenAIMessage::user_or_assistant(role, text))
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Main chat handler — agentic loop
    // -----------------------------------------------------------------------

    fn handle_chat(&self, session_id: &str, payload: &serde_json::Value) {
        let base_url = match Self::require_default("openaiCompatibleBaseURL") {
            Ok(v) => v,
            Err(e) => {
                push_ai_event(
                    session_id,
                    &AIEvent::Error {
                        message: format!("{:?}", e),
                    },
                );
                push_ai_event(
                    session_id,
                    &AIEvent::TurnCompleted {
                        session_id: session_id.to_string(),
                    },
                );
                return;
            }
        };

        let api_key = get_user_default("openaiCompatibleAPIKey").unwrap_or_default();
        let model = get_user_default("openaiCompatibleModel")
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "gpt-4o".to_string());

        let initial = payload
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|v| Self::to_openai_messages(v))
            .unwrap_or_default();

        if initial.is_empty() {
            push_ai_event(
                session_id,
                &AIEvent::Error {
                    message: "No messages in payload".into(),
                },
            );
            push_ai_event(
                session_id,
                &AIEvent::TurnCompleted {
                    session_id: session_id.to_string(),
                },
            );
            return;
        }

        log!(
            "[OpenAI Plugin] starting agentic loop model={} messages={}",
            model,
            initial.len()
        );

        let mut messages = initial;
        let max_rounds = 10;

        for round in 0..max_rounds {
            log!("[OpenAI Plugin] round {} of {}", round + 1, max_rounds);

            let result = Self::stream_one_round(session_id, &base_url, &api_key, &model, &messages);

            match result.finish_reason {
                FinishReason::ToolCalls if !result.tool_calls.is_empty() => {
                    // Append the assistant's tool-call message to history.
                    let assistant_calls: Vec<AssistantToolCall> = result
                        .tool_calls
                        .iter()
                        .map(|tc| AssistantToolCall {
                            id: tc.id.clone(),
                            kind: "function".into(),
                            function: AssistantToolCallFunction {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect();
                    messages.push(OpenAIMessage::assistant_tool_calls(assistant_calls));

                    // Execute each tool and append results.
                    for tc in &result.tool_calls {
                        log!(
                            "[OpenAI Plugin] executing tool '{}' args={}",
                            tc.name,
                            &tc.arguments[..tc.arguments.len().min(120)]
                        );
                        let tool_result = Self::execute_tool(&tc.name, &tc.arguments);
                        log!(
                            "[OpenAI Plugin] tool '{}' result_len={}",
                            tc.name,
                            tool_result.len()
                        );
                        messages.push(OpenAIMessage::tool_result(tc.id.clone(), tool_result));
                    }
                    // Continue to next round.
                }
                _ => {
                    // stop, unknown, or tool_calls with empty list — we're done.
                    break;
                }
            }
        }

        log!("[OpenAI Plugin] agentic loop complete, emitting TurnCompleted");
        push_ai_event(
            session_id,
            &AIEvent::TurnCompleted {
                session_id: session_id.to_string(),
            },
        );
    }
}

// Helper — can't add inherent methods to OpenAIMessage from outside impl block,
// so use a free function for the conversion case.
impl OpenAIMessage {
    fn user_or_assistant(role: &str, text: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin entry points
// ---------------------------------------------------------------------------

#[chorograph_plugin]
pub fn init() {
    let base_url = get_user_default("openaiCompatibleBaseURL").unwrap_or_default();
    let api_key = get_user_default("openaiCompatibleAPIKey").unwrap_or_default();
    let saved_model = get_user_default("openaiCompatibleModel").unwrap_or_default();

    let model_component = if !base_url.is_empty() {
        match OpenAIPlugin::fetch_models(&base_url, &api_key) {
            Ok(models) if !models.is_empty() => {
                let selected = if models.contains(&saved_model) {
                    saved_model.clone()
                } else {
                    models[0].clone()
                };
                let options: Vec<serde_json::Value> = models
                    .iter()
                    .map(|id| json!({ "value": id, "label": id }))
                    .collect();
                json!({
                    "type": "picker",
                    "id": "model",
                    "label": "Model",
                    "options": options,
                    "selected": selected,
                    "defaults_key": "openaiCompatibleModel"
                })
            }
            _ => json!({
                "type": "textfield",
                "id": "model",
                "label": "Model",
                "placeholder": "gpt-4o",
                "value": saved_model,
                "defaults_key": "openaiCompatibleModel"
            }),
        }
    } else {
        json!({
            "type": "textfield",
            "id": "model",
            "label": "Model",
            "placeholder": "gpt-4o",
            "value": saved_model,
            "defaults_key": "openaiCompatibleModel"
        })
    };

    let ui = json!([
        { "type": "label", "text": "OpenAI Compatible" },
        {
            "type": "textfield",
            "id": "baseUrl",
            "label": "Base URL",
            "placeholder": "http://localhost:11434",
            "value": base_url,
            "defaults_key": "openaiCompatibleBaseURL"
        },
        {
            "type": "securefield",
            "id": "apiKey",
            "label": "API Key",
            "placeholder": "sk-... (leave blank for local models)",
            "value": "",
            "defaults_key": "openaiCompatibleAPIKey"
        },
        model_component
    ]);
    push_ui(&ui.to_string());
}

#[chorograph_plugin]
pub fn handle_action(action_id: String, payload: serde_json::Value) {
    let plugin = OpenAIPlugin;

    log!(
        "[OpenAI Plugin] handle_action id={} payload_keys={:?}",
        action_id,
        payload.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    if action_id == "chat" || action_id == "reply" {
        if let Some(session_id) = payload.get("session_id").and_then(|s| s.as_str()) {
            plugin.handle_chat(session_id, &payload);
        }
    } else if action_id == "engage" {
        if let (Some(session_id), Some(prompt)) = (
            payload.get("session_id").and_then(|s| s.as_str()),
            payload.get("prompt").and_then(|p| p.as_str()),
        ) {
            let synthetic = json!({
                "session_id": session_id,
                "messages": [{ "role": "user", "text": prompt }]
            });
            plugin.handle_chat(session_id, &synthetic);
        }
    }
}
