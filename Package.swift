// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ChorographOpenAIPlugin",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "ChorographOpenAIPlugin",
            type: .dynamic,
            targets: ["ChorographOpenAIPlugin"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/aorgcorn/chorograph-plugin-sdk.git",
            from: "1.0.0"
        ),
    ],
    targets: [
        .target(
            name: "ChorographOpenAIPlugin",
            dependencies: [
                .product(name: "ChorographPluginSDK", package: "chorograph-plugin-sdk"),
            ],
            linkerSettings: [
                // Ensure the plugin resolves libChorographPluginSDK.dylib from
                // the host process rather than a bundled copy.
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"]),
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"]),
            ]
        ),
    ]
)
