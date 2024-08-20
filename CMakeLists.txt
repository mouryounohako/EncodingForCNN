cmake_minimum_required(VERSION 3.13)

project(example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -pthread)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -fPIC")

message(STATUS "The CXX compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "The C   compiler : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/../cxx_sdk_v2 ${CMAKE_SOURCE_DIR}/../c_sdk_v2)
link_directories(${CMAKE_SOURCE_DIR}/../lib)

add_subdirectory(bolt)
add_subdirectory(neujeans)
add_subdirectory(bolt_fpga)
add_subdirectory(neujeans_fpga)
add_subdirectory(res_nj_fpga)
add_subdirectory(res_plain)
add_subdirectory(tests)
