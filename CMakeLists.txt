cmake_minimum_required(VERSION 3.15)
project(YOLOV5_DETECTION LANGUAGES CXX CUDA)                # 项目信息

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 11)                         # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)                # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 11)                          # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON)                 # 表明如果找不到指定版本的 C++ 编译器，将发出错误
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)   # 设置NVCC编译器

# TensorRT
set(TENSORRT_ROOT /home/sang/packages/TensorRT-8.5.1.7)
include_directories(${TENSORRT_ROOT}/include/)
link_directories(${TENSORRT_ROOT}/lib)

# CUDA
set(CUDA_ROOT /usr/local/cuda)
include_directories(${CUDA_ROOT}/include/)
link_directories(${CUDA_ROOT}/lib64)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

# gflags
#include_directories(${gflags_INCLUDE_DIRS})
#link_libraries(${gflags_LIBRARIES})
option(ENABLE_DEBUG "Enable Debug" OFF)
if (ENABLE_DEBUG)
    message("++ Compile and debug versions of the program.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -g -arch=sm_86")  # 调试版
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    message("++ Compile and release versions of the program.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86")         # 发布版
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 添加可执行文件
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/cpp/*.cpp ${PROJECT_SOURCE_DIR}/src/cpp/*.cu)
set(INCS ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_executable(trt-infer ${SRCS})
target_include_directories(trt-infer PRIVATE ${INCS}) 

target_link_libraries(trt-infer PRIVATE yaml-cpp)
target_link_libraries(trt-infer PRIVATE -lstdc++fs) # 编译experimental/filesystem模块
target_link_libraries(trt-infer PRIVATE nvinfer nvonnxparser nvparsers nvinfer_plugin)  # 链接TensorRT动态库
target_link_libraries(trt-infer PRIVATE cudart)                                         # 链接CUDA动态库

set_target_properties(trt-infer PROPERTIES CUDA_ARCHITECTURES "86") # 假设你的目标架构是Compute Capability 8.6

