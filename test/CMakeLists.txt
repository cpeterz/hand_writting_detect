cmake_minimum_required(VERSION 3.1)

project(Test)

find_package(OpenCV REQUIRED)

add_executable(test test.cpp)

target_link_libraries(test ${OpenCV_LIBS})