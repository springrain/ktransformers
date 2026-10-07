cmake_minimum_required(VERSION 3.16)
project(kt_gec_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test_gec test_gec.cpp)
target_include_directories(test_gec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

enable_testing()
add_test(NAME gec COMMAND test_gec)
