// SettingsView.swift — OpenAI Compatible settings panel

import SwiftUI
import ChorographPluginSDK

struct OpenAICompatibleSettingsView: View {
    @AppStorage("openaiCompatibleBaseURL") private var baseURL = "http://localhost:11434"
    @AppStorage("openaiCompatibleAPIKey")  private var apiKey  = ""
    @AppStorage("openaiCompatibleModel")   private var selectedModel = ""

    @State private var healthStatus: HealthStatus = .unknown
    @State private var availableModels: [ProviderModel] = []
    @State private var isLoadingModels = false

    private let provider = OpenAICompatibleProvider()

    enum HealthStatus {
        case unknown, checking, ok(String), failed(String)
        var isChecking: Bool { if case .checking = self { return true }; return false }
        var label: String {
            switch self {
            case .unknown:         return "Not checked"
            case .checking:        return "Checking…"
            case .ok(let v):       return v.isEmpty ? "Connected" : v
            case .failed(let msg): return msg
            }
        }
        var color: Color {
            switch self {
            case .unknown, .checking: return .secondary
            case .ok:                 return .green
            case .failed:             return .red
            }
        }
    }

    var body: some View {
        Form {
            Section("Connection") {
                TextField("Base URL", text: $baseURL)
                    .onSubmit { Task { await checkHealth() } }

                SecureField("API Key (leave blank for local models)", text: $apiKey)
                    .onSubmit { Task { await checkHealth() } }

                HStack {
                    Button("Check") { Task { await checkHealth() } }
                        .buttonStyle(.borderedProminent)
                        .disabled(healthStatus.isChecking)

                    if healthStatus.isChecking { ProgressView().scaleEffect(0.7) }

                    Text(healthStatus.label)
                        .font(.caption)
                        .foregroundStyle(healthStatus.color)
                }
            }

            Section("Model") {
                if isLoadingModels {
                    HStack {
                        ProgressView().scaleEffect(0.7)
                        Text("Loading models…").font(.caption).foregroundStyle(.secondary)
                    }
                } else if availableModels.isEmpty {
                    Text("No models available. Check your connection and click Refresh.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $selectedModel) {
                        Text("Provider default").tag("")
                        ForEach(availableModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
                    }
                    .onChange(of: selectedModel) { _, newValue in
                        Task { await provider.setSelectedModel(newValue.isEmpty ? nil : newValue) }
                    }
                }

                Button("Refresh models") { Task { await loadModels() } }
                    .disabled(isLoadingModels)
            }

            Section("Info") {
                Text("Compatible with OpenAI, Ollama, LM Studio, vLLM, and any server that implements the OpenAI chat completions API.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Default port for Ollama is 11434. For OpenAI, use https://api.openai.com")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding()
        .task { await checkHealth() }
        .task { await loadModels() }
    }

    private func checkHealth() async {
        healthStatus = .checking
        let h = await provider.health()
        if h.isReachable {
            healthStatus = .ok(h.version ?? "")
        } else {
            healthStatus = .failed(h.detail ?? "Not reachable")
        }
    }

    private func loadModels() async {
        isLoadingModels = true
        do {
            availableModels = try await provider.availableModels()
        } catch {
            availableModels = []
        }
        isLoadingModels = false
    }
}
