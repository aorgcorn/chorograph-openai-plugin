use chorograph_plugin_sdk_rust::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// OpenAI-compatible REST API plugin (WASM path)
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

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

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

    /// POST to /v1/chat/completions and return the assistant reply text.
    fn call_chat_completions(
        base_url: &str,
        api_key: &str,
        model: &str,
        messages: Vec<OpenAIMessage>,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

        let body = serde_json::to_string(&json!({
            "model": model,
            "messages": messages,
            "stream": false
        }))
        .map_err(|e| PluginError::SerializationError(e.to_string()))?;

        let mut headers: Vec<(&str, String)> = vec![
            ("Content-Type", "application/json".to_string()),
            ("Accept", "application/json".to_string()),
        ];
        let auth_value;
        if let Some(auth) = Self::auth_header(api_key) {
            auth_value = auth;
            headers.push(("Authorization", auth_value.clone()));
        }

        let header_refs: Vec<(&str, &str)> =
            headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let resp = http_post(&url, Some(&header_refs), &body)?;

        if resp.status < 200 || resp.status >= 300 {
            return Err(PluginError::Other(format!(
                "OpenAI API returned HTTP {}: {}",
                resp.status, resp.body
            )));
        }

        let parsed: OpenAIResponse = serde_json::from_str(&resp.body).map_err(|e| {
            PluginError::SerializationError(format!("Failed to parse response: {}", e))
        })?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| PluginError::Other("OpenAI response contained no choices".to_string()))
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
            "[OpenAI Plugin] Calling {} model={} messages={}",
            base_url,
            model,
            messages.len()
        );

        match Self::call_chat_completions(&base_url, &api_key, &model, messages) {
            Ok(text) => {
                push_ai_event(
                    session_id,
                    &AIEvent::StreamingDelta {
                        session_id: session_id.to_string(),
                        text,
                    },
                );
            }
            Err(e) => {
                push_ai_event(
                    session_id,
                    &AIEvent::Error {
                        message: format!("OpenAI request failed: {:?}", e),
                    },
                );
            }
        }

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
    // Pre-populate fields from previously saved UserDefaults values.
    let base_url = get_user_default("openaiCompatibleBaseURL").unwrap_or_default();
    let model = get_user_default("openaiCompatibleModel").unwrap_or_default();
    // API key is intentionally not pre-populated (securefield shows blank for security).

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
        {
            "type": "textfield",
            "id": "model",
            "label": "Model",
            "placeholder": "gpt-4o",
            "value": model,
            "defaults_key": "openaiCompatibleModel"
        }
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
