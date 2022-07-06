cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaOpenGL LANGUAGES CXX CUDA) # enable CUDA support 

set(CMAKE_CUDA_ARCHITECTURES 35)

include_directories(libs/glew-2.1.0/include)
include_directories(libs/freeglut/include)

link_directories(libs/glew-2.1.0/lib/Release/x64)
link_directories(libs/freeglut/lib/x64)

add_executable(kernel kernel.cu)
target_link_libraries(kernel glew32 freeglut)