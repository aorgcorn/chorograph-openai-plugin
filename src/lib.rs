use chorograph_plugin_sdk_rust::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// OpenAI-compatible REST API plugin (WASM path) — streaming via SSE
// ---------------------------------------------------------------------------
// Reads configuration from the host UserDefaults via `get_user_default`:
//   "openaiCompatibleBaseURL"  — base URL, e.g. "http://localhost:11434"
//   "openaiCompatibleAPIKey"   — API key (may be empty for local models)
//   "openaiCompatibleModel"    — model ID, e.g. "gpt-4o" or "llama3"
// ---------------------------------------------------------------------------

struct OpenAIPlugin;

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

// ---------------------------------------------------------------------------
// Streaming response types (for `"stream": true`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
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

impl OpenAIPlugin {
    /// Read a required config value from UserDefaults.
    /// Returns `Err` with a human-readable message if the key is unset.
    fn require_default(key: &str) -> Result<String> {
        get_user_default(key).ok_or_else(|| {
            PluginError::Other(format!(
                "OpenAI plugin: '{}' is not configured. Please set it in Settings.",
                key
            ))
        })
    }

    /// Build the Authorization header value (empty string → no auth header).
    fn auth_header(api_key: &str) -> Option<String> {
        if api_key.is_empty() {
            None
        } else {
            Some(format!("Bearer {}", api_key))
        }
    }

    /// Stream a chat completion, emitting one `StreamingDelta` per token.
    ///
    /// Uses `"stream": true` and the SSE host API.  Emits a `ToolCall` log
    /// line before the request so the activity panel shows what's happening.
    fn stream_chat_completions(
        session_id: &str,
        base_url: &str,
        api_key: &str,
        model: &str,
        messages: Vec<OpenAIMessage>,
    ) {
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

        let body = match serde_json::to_string(&json!({
            "model": model,
            "messages": messages,
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
                return;
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

        // Log what we're about to call.
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
                return;
            }
        };

        // Iterate over SSE lines until [DONE].
        let sid = session_id.to_string();
        for_each_sse_line(handle, move |line| {
            // SSE lines look like:  data: <json>
            // or the terminator:    data: [DONE]
            let data = match line.strip_prefix("data: ") {
                Some(d) => d.trim(),
                None => return true, // skip comment/event/retry lines
            };

            if data == "[DONE]" {
                return false; // stop iterating
            }

            if data.is_empty() {
                return true;
            }

            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                Ok(chunk) => {
                    if let Some(choice) = chunk.choices.into_iter().next() {
                        if let Some(text) = choice.delta.content {
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
                    }
                }
                Err(e) => {
                    // Log but don't abort — some providers send non-JSON lines.
                    log!(
                        "[OpenAI Plugin] failed to parse SSE chunk: {} — {:?}",
                        data,
                        e
                    );
                }
            }

            true // continue
        });
        log!("[OpenAI Plugin] SSE loop done");
    }

    /// GET /v1/models and return model IDs.
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

    /// Convert Chorograph ChatMessage array (role/text) to OpenAI format (role/content).
    fn to_openai_messages(messages: &[serde_json::Value]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .filter_map(|m| {
                let role = m.get("role").and_then(|r| r.as_str())?;
                let text = m.get("text").and_then(|t| t.as_str()).unwrap_or("");
                Some(OpenAIMessage {
                    role: role.to_string(),
                    content: text.to_string(),
                })
            })
            .collect()
    }

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

        let messages = payload
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|v| Self::to_openai_messages(v))
            .unwrap_or_default();

        if messages.is_empty() {
            push_ai_event(
                session_id,
                &AIEvent::Error {
                    message: "No messages in payload".to_string(),
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
            "[OpenAI Plugin] Calling {} model={} messages={} (streaming)",
            base_url,
            model,
            messages.len()
        );

        Self::stream_chat_completions(session_id, &base_url, &api_key, &model, messages);

        log!(
            "[OpenAI Plugin] stream complete, emitting TurnCompleted sid={}",
            &session_id[..session_id.len().min(8)]
        );
        push_ai_event(
            session_id,
            &AIEvent::TurnCompleted {
                session_id: session_id.to_string(),
            },
        );
    }
}

#[chorograph_plugin]
pub fn init() {
    let base_url = get_user_default("openaiCompatibleBaseURL").unwrap_or_default();
    let api_key = get_user_default("openaiCompatibleAPIKey").unwrap_or_default();
    let saved_model = get_user_default("openaiCompatibleModel").unwrap_or_default();

    // Try to populate the model picker from /v1/models if a base URL is configured.
    let model_component = if !base_url.is_empty() {
        match OpenAIPlugin::fetch_models(&base_url, &api_key) {
            Ok(models) if !models.is_empty() => {
                // Use the saved model as the selected value; fall back to the first model.
                let selected = if models.contains(&saved_model) {
                    saved_model.clone()
                } else {
                    models[0].clone()
                };
                // Persist the resolved selection so handle_chat picks it up immediately.
                // (Only update UserDefaults when the saved value is absent or stale.)
                if saved_model.is_empty() || !models.contains(&saved_model) {
                    // Note: we can only write via the defaults_key mechanism on user
                    // interaction; just keep the saved_model unchanged and let the
                    // picker selection drive writes on first user change.
                }
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
            _ => {
                // Fetch failed or returned empty — fall back to free-text entry.
                json!({
                    "type": "textfield",
                    "id": "model",
                    "label": "Model",
                    "placeholder": "gpt-4o",
                    "value": saved_model,
                    "defaults_key": "openaiCompatibleModel"
                })
            }
        }
    } else {
        // No base URL yet — plain textfield.
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
        // Legacy single-prompt action
        if let (Some(session_id), Some(prompt)) = (
            payload.get("session_id").and_then(|s| s.as_str()),
            payload.get("prompt").and_then(|p| p.as_str()),
        ) {
            let synthetic_payload = json!({
                "session_id": session_id,
                "messages": [{ "role": "user", "text": prompt }]
            });
            plugin.handle_chat(session_id, &synthetic_payload);
        }
    }
}
