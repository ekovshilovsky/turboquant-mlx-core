// swift-tools-version: 5.9
import PackageDescription

// The C++ core library (libturboquant_mlx.dylib) is built via CMake.
// SPM only needs to expose the pure-C API header for SwiftLM to import;
// linking against the pre-built dylib is handled by the consumer (SwiftLM).
let package = Package(
    name: "TurboQuantMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "TurboQuantC", targets: ["TurboQuantC"]),
    ],
    targets: [
        .target(
            name: "TurboQuantC",
            path: "include/turboquant_c",
            publicHeadersPath: "."
        ),
    ]
)
