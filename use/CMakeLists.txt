cmake_minimum_required(VERSION 3.1)

project(Use)

find_package(OpenCV 3.4.13 REQUIRED)

add_executable(use use.cpp)

target_link_libraries(use ${OpenCV_LIBS})