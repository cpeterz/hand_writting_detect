cmake_minimum_required(VERSION 3.1)

project(Cutting)

find_package(OpenCV REQUIRED)

add_executable(cutting cutting.cpp)

target_link_libraries(cutting ${OpenCV_LIBS})