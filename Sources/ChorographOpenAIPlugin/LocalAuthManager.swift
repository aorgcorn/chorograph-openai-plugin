// LocalAuthManager.swift
// Validates connectivity to an OpenAI-compatible API endpoint by calling GET /v1/models.

import Foundation

actor LocalAuthManager {

    struct ValidationResult: Sendable {
        let isValid: Bool
        let version: String?
        let errorMessage: String?
    }

    func validate(baseURL: String, apiKey: String) async -> ValidationResult {
        guard !baseURL.isEmpty,
              let url = URL(string: baseURL.trimmingCharacters(in: .whitespaces)
                              .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                           + "/v1/models") else {
            return ValidationResult(
                isValid: false,
                version: nil,
                errorMessage: "Invalid base URL: '\(baseURL)'"
            )
        }

        var request = URLRequest(url: url, timeoutInterval: 10)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0

            if statusCode == 401 {
                return ValidationResult(
                    isValid: false,
                    version: nil,
                    errorMessage: "Authentication failed (401) — check your API key."
                )
            }

            if statusCode < 200 || statusCode >= 300 {
                return ValidationResult(
                    isValid: false,
                    version: nil,
                    errorMessage: "Server returned HTTP \(statusCode)."
                )
            }

            // Try to parse the model count as the "version" info shown in the UI
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let modelData = json["data"] as? [[String: Any]] {
                let count = modelData.count
                return ValidationResult(
                    isValid: true,
                    version: "\(count) model\(count == 1 ? "" : "s") available",
                    errorMessage: nil
                )
            }

            return ValidationResult(isValid: true, version: nil, errorMessage: nil)
        } catch {
            return ValidationResult(
                isValid: false,
                version: nil,
                errorMessage: "Connection failed: \(error.localizedDescription)"
            )
        }
    }
}
