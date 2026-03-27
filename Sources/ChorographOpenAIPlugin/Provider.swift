// Provider.swift — OpenAICompatibleProvider
// AIProvider implementation that calls any OpenAI-compatible REST API
// (OpenAI, Ollama, LM Studio, vLLM, etc.) directly via URLSession.

import Foundation
import ChorographPluginSDK

actor OpenAICompatibleProvider: AIProvider {

    // MARK: - Identity

    nonisolated let id: ProviderID = "openai-compatible"
    nonisolated let displayName: String = "OpenAI Compatible"
    nonisolated let supportsSymbolSearch: Bool = false

    // MARK: - Configuration (live UserDefaults reads)

    var baseURL: String {
        UserDefaults.standard.string(forKey: "openaiCompatibleBaseURL") ?? "http://localhost:11434"
    }

    var apiKey: String {
        UserDefaults.standard.string(forKey: "openaiCompatibleAPIKey") ?? ""
    }

    var selectedModel: String? {
        get { UserDefaults.standard.string(forKey: "openaiCompatibleModel") }
        set { UserDefaults.standard.set(newValue, forKey: "openaiCompatibleModel") }
    }

    // MARK: - Internal state

    private let localAuth: LocalAuthManager
    private var eventContinuation: AsyncStream<any ProviderEvent>.Continuation?
    private var isStopped = false
    private var sessionResults: [String: String] = [:]

    // MARK: - Init

    init(localAuth: LocalAuthManager = LocalAuthManager()) {
        self.localAuth = localAuth
    }

    // MARK: - Health

    func health() async -> ProviderHealth {
        let result = await localAuth.validate(baseURL: baseURL, apiKey: apiKey)
        return ProviderHealth(
            isReachable: result.isValid,
            version: result.version,
            detail: result.errorMessage,
            activeModel: selectedModel
        )
    }

    // MARK: - Sessions

    func createSession(title: String?) async throws -> ProviderSession {
        let id = UUID().uuidString
        sessionResults[id] = ""
        return ProviderSession(id: id, title: title)
    }

    func sendMessage(sessionID: String, text: String) async throws {
        let url = baseURL
        let key = apiKey
        let model = selectedModel ?? "gpt-4o"
        let continuation = self.eventContinuation

        Task {
            await self.runChatRequest(
                sessionID: sessionID,
                messages: [["role": "user", "content": text]],
                baseURL: url,
                apiKey: key,
                model: model,
                continuation: continuation
            )
        }
    }

    func abortSession(id: String) async throws {
        eventContinuation?.yield(TurnFinishedEvent(sessionID: id))
    }

    func fetchLastAssistantText(sessionID: String) async throws -> String {
        sessionResults[sessionID] ?? ""
    }

    func availableModels() async throws -> [ProviderModel] {
        let url = baseURL
        let key = apiKey

        guard let requestURL = URL(string: url.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1/models") else {
            return []
        }

        var request = URLRequest(url: requestURL, timeoutInterval: 10)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }

        let (data, _) = try await URLSession.shared.data(for: request)

        struct ModelsResponse: Decodable {
            struct Model: Decodable { let id: String }
            let data: [Model]
        }

        let parsed = try JSONDecoder().decode(ModelsResponse.self, from: data)
        return parsed.data.map { ProviderModel(id: $0.id, displayName: $0.id) }
    }

    func setSelectedModel(_ modelID: String?) {
        selectedModel = modelID
    }

    func setShimEnvironment(socketPath: String, shimDirPath: String) {
        // Not applicable — HTTP-based provider has no subprocess shim.
    }

    // MARK: - Event stream

    func eventStream() -> AsyncStream<any ProviderEvent> {
        isStopped = false
        var capturedCont: AsyncStream<any ProviderEvent>.Continuation?
        let stream = AsyncStream<any ProviderEvent> { cont in
            capturedCont = cont
        }
        self.eventContinuation = capturedCont
        capturedCont?.yield(ConnectedEvent())
        return stream
    }

    func stopEventStream() {
        isStopped = true
        eventContinuation?.finish()
        eventContinuation = nil
    }

    // MARK: - HTTP request

    private func runChatRequest(
        sessionID: String,
        messages: [[String: String]],
        baseURL: String,
        apiKey: String,
        model: String,
        continuation: AsyncStream<any ProviderEvent>.Continuation?
    ) async {
        let urlString = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1/chat/completions"
        guard let url = URL(string: urlString) else {
            continuation?.yield(ErrorEvent("Invalid base URL: \(baseURL)"))
            continuation?.yield(TurnFinishedEvent(sessionID: sessionID))
            return
        }

        var request = URLRequest(url: url, timeoutInterval: 60)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let body: [String: Any] = [
            "model": model,
            "messages": messages,
            "stream": false
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            continuation?.yield(ErrorEvent("Failed to encode request: \(error.localizedDescription)"))
            continuation?.yield(TurnFinishedEvent(sessionID: sessionID))
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0

            if statusCode < 200 || statusCode >= 300 {
                let body = String(data: data, encoding: .utf8) ?? "<no body>"
                continuation?.yield(ErrorEvent("API error \(statusCode): \(body)"))
                continuation?.yield(TurnFinishedEvent(sessionID: sessionID))
                return
            }

            struct Choice: Decodable {
                struct Message: Decodable { let content: String }
                let message: Message
            }
            struct ChatResponse: Decodable { let choices: [Choice] }

            let parsed = try JSONDecoder().decode(ChatResponse.self, from: data)
            let text = parsed.choices.first?.message.content ?? ""

            sessionResults[sessionID] = text
            continuation?.yield(AssistantReplyEvent(sessionID: sessionID, text: text))
            continuation?.yield(TurnFinishedEvent(sessionID: sessionID))
        } catch {
            continuation?.yield(ErrorEvent("Request failed: \(error.localizedDescription)"))
            continuation?.yield(TurnFinishedEvent(sessionID: sessionID))
        }
    }
}
