cmake_minimum_required(VERSION 3.10)
project(parallel)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

add_executable(parallel main.cpp thread_num.h barrier.h reduce_par.h)
target_link_libraries(parallel PUBLIC OpenMP::OpenMP_CXX pthread)
