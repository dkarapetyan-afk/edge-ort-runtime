cmake_minimum_required(VERSION 3.22.1)
project(edge_ort_android)

set(CMAKE_CXX_STANDARD 17)

# Fetch or locate Google Oboe
find_package(oboe REQUIRED CONFIG)

# Include directories
include_directories(
    cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../target/${ANDROID_ABI}/release
)

# Imported prebuilt Rust cdylib
add_library(edge_ort_runtime SHARED IMPORTED)
set_target_properties(edge_ort_runtime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libedge_ort_runtime.so"
)

# Native audio capture wrapper
add_library(edge_ort_oboe SHARED
    cpp/OboeAudioCapture.cpp
)

target_link_libraries(edge_ort_oboe
    oboe::oboe
    edge_ort_runtime
    log
)
