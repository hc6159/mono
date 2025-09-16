cmake_minimum_required(VERSION 3.16)
project(IntegratedDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIRS})

# Source files
set(SOURCES
    pose_detection.cpp
    integrated_detection.cpp
    main_integrated.cpp
)

# Headers
set(HEADERS
    pose_detection.h
    integrated_detection.h
)

# Create executable
add_executable(integrated_detection ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(integrated_detection 
    ${OpenCV_LIBS}
    onnxruntime
)

# Compiler flags
target_compile_options(integrated_detection PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# Set C++ standard
set_target_properties(integrated_detection PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime found: ${onnxruntime_FOUND}")

# Optional: Add CUDA support if available
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    target_compile_definitions(integrated_detection PRIVATE ENABLE_CUDA)
endif()

# Optional: Add TensorRT support if available
find_package(TensorRT QUIET)
if(TensorRT_FOUND)
    message(STATUS "TensorRT found")
    target_compile_definitions(integrated_detection PRIVATE ENABLE_TENSORRT)
endif()