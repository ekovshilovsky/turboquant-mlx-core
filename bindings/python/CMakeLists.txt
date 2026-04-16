cmake_minimum_required(VERSION 3.24)
project(turboquant_mlx_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(MLX REQUIRED CONFIG)

# Import the pre-built TurboQuant core shared library from the repository build directory
set(TQ_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(tq_core SHARED IMPORTED)
set_target_properties(tq_core PROPERTIES
    IMPORTED_LOCATION "${TQ_CORE_DIR}/build/libturboquant_mlx.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${TQ_CORE_DIR}/include"
)

nanobind_add_module(turboquant_mlx bindings.cpp)
target_link_libraries(turboquant_mlx PRIVATE tq_core mlx)

# Embed rpath so the extension can locate libturboquant_mlx.dylib at runtime
set_target_properties(turboquant_mlx PROPERTIES
    BUILD_RPATH "${TQ_CORE_DIR}/build"
    INSTALL_RPATH "${TQ_CORE_DIR}/build"
)

install(TARGETS turboquant_mlx LIBRARY DESTINATION .)
