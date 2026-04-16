cmake_minimum_required(VERSION 3.24)
project(turboquant_mlx_core VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX (installed via Homebrew or custom path)
# MLXConfig.cmake includes extension.cmake which provides mlx_build_metallib()
find_package(MLX REQUIRED CONFIG)

# Core C++ library
add_library(turboquant_mlx SHARED
  src/codebook.cpp
  src/rotation.cpp
  src/quantizer.cpp
  src/dequantizer.cpp
  src/linear.cpp
  src/converter.cpp
  src/distributed.cpp
  src/kv_cache.cpp
  src/decode_buffer.cpp
  src/attention_dispatch.cpp
  src/serialization.cpp
  src/turboquant_c.cpp
  src/turboquant_c_dequant.cpp
)
target_include_directories(turboquant_mlx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(turboquant_mlx PUBLIC mlx)
target_compile_options(turboquant_mlx PRIVATE ${MLX_CXX_FLAGS})

# Metal shader library
mlx_build_metallib(
  TARGET turboquant_mlx_metallib
  TITLE turboquant_mlx
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_dequant.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_dequant_mm.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_wht.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_quantize.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_attention.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/turboquant_qjl.metal
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
  OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_dependencies(turboquant_mlx turboquant_mlx_metallib)

# CLI tools
add_executable(tq-convert tools/tq_convert.cpp)
target_link_libraries(tq-convert PRIVATE turboquant_mlx)

add_executable(tq-dequant tools/tq_dequant_model.cpp)
target_link_libraries(tq-dequant PRIVATE turboquant_mlx)

# Tests
option(TQ_BUILD_TESTS "Build tests" ON)
if(TQ_BUILD_TESTS)
  enable_testing()

  # Unit tests — each file has its own main(), build separately
  file(GLOB TQ_UNIT_TEST_SOURCES tests/unit/*.cpp)
  foreach(TEST_SOURCE ${TQ_UNIT_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(tq_${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(tq_${TEST_NAME} PRIVATE turboquant_mlx)
    add_test(NAME unit_${TEST_NAME} COMMAND tq_${TEST_NAME})
  endforeach()

  # Integration tests — each file has its own main(), build separately
  file(GLOB TQ_INTEGRATION_TEST_SOURCES tests/integration/*.cpp)
  foreach(TEST_SOURCE ${TQ_INTEGRATION_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(tq_${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(tq_${TEST_NAME} PRIVATE turboquant_mlx)
    add_test(NAME integration_${TEST_NAME} COMMAND tq_${TEST_NAME})
  endforeach()

  # C API tests (compiled as C, not C++) — each file has its own main()
  file(GLOB TQ_C_API_TEST_SOURCES tests/c_api/*.c)
  foreach(TEST_SOURCE ${TQ_C_API_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(tq_${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(tq_${TEST_NAME} PRIVATE turboquant_mlx)
    set_target_properties(tq_${TEST_NAME} PROPERTIES LINKER_LANGUAGE C)
    add_test(NAME c_api_${TEST_NAME} COMMAND tq_${TEST_NAME})
  endforeach()
endif()

# Benchmarks
option(TQ_BUILD_BENCHMARKS "Build benchmarks" OFF)
if(TQ_BUILD_BENCHMARKS)
  file(GLOB TQ_BENCH_SOURCES benchmarks/*.cpp)
  add_executable(tq_benchmarks ${TQ_BENCH_SOURCES})
  target_link_libraries(tq_benchmarks PRIVATE turboquant_mlx)
endif()

# Binding IDE targets — provide include paths for clangd analysis without
# compiling. The actual builds use pip (Python) and npm (Node.js).
option(TQ_BUILD_BINDING_IDE_TARGETS "Create IDE analysis targets for bindings" ON)
if(TQ_BUILD_BINDING_IDE_TARGETS)
  # Python bindings: nanobind headers from the project venv
  set(TQ_NANOBIND_INCLUDE "" CACHE PATH "Path to nanobind include dir (e.g., .venv/.../nanobind/include)")
  set(TQ_PYTHON_INCLUDE "" CACHE PATH "Path to Python.h include dir")
  if(EXISTS "${TQ_NANOBIND_INCLUDE}/nanobind/nanobind.h")
    add_library(tq_python_bindings_ide OBJECT bindings/python/bindings.cpp)
    target_link_libraries(tq_python_bindings_ide PRIVATE turboquant_mlx)
    target_include_directories(tq_python_bindings_ide PRIVATE
      ${TQ_NANOBIND_INCLUDE}
      ${TQ_PYTHON_INCLUDE}
    )
    set_target_properties(tq_python_bindings_ide PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()

  # Node.js bindings: node_api.h from the system Node installation
  find_path(NODE_INCLUDE_DIR node_api.h
    HINTS
      /opt/homebrew/include/node
      /usr/local/include/node
      $ENV{NODE_PATH}/include
  )
  if(NODE_INCLUDE_DIR)
    add_library(tq_node_bindings_ide OBJECT bindings/node/addon.cpp)
    target_link_libraries(tq_node_bindings_ide PRIVATE turboquant_mlx)
    target_include_directories(tq_node_bindings_ide PRIVATE ${NODE_INCLUDE_DIR})
    set_target_properties(tq_node_bindings_ide PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endif()
