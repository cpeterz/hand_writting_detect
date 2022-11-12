cmake_minimum_required(VERSION 3.1)

project(Train)

find_package(OpenCV REQUIRED)

add_executable(train train.cpp)

target_link_libraries(train ${OpenCV_LIBS})