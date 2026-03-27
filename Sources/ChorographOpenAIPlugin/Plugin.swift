// Plugin.swift — ChorographOpenAIPlugin
// Entry point for the OpenAI-compatible Chorograph plugin.
// Registers OpenAICompatibleProvider as an AI provider and a settings panel.

import ChorographPluginSDK
import SwiftUI

public final class OpenAICompatiblePlugin: ChorographPlugin, @unchecked Sendable {

    public let manifest = PluginManifest(
        id: "com.chorograph.plugin.openai-compatible",
        displayName: "OpenAI Compatible",
        description: "Connects to any OpenAI-compatible API (OpenAI, Ollama, LM Studio, vLLM, etc.).",
        version: "1.0.0",
        capabilities: [.aiProvider, .settingsPanel]
    )

    public init() {}

    public func bootstrap(context: any PluginContextProviding) async throws {
        context.registerProvider(OpenAICompatibleProvider())
        context.registerSettingsPanel(title: "OpenAI Compatible", AnyView(OpenAICompatibleSettingsView()))
    }
}

// MARK: - C-ABI factory (required for dlopen-based loading)

@_cdecl("chorograph_plugin_create")
public func chorographPluginCreate() -> UnsafeMutableRawPointer {
    let plugin = OpenAICompatiblePlugin()
    return Unmanaged.passRetained(plugin as AnyObject).toOpaque()
}
