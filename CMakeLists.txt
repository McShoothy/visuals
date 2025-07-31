cmake_minimum_required(VERSION 3.20)
project(stage)

set(CMAKE_CXX_STANDARD 23)

add_executable(stage main.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(stage ${OpenCV_LIBS})
