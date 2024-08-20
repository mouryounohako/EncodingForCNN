list(APPEND CMAKE_MODULE_PATH "/home/linuxbrew/install/EzPC/SCI/build/install/lib/cmake/SCI")
set(CMAKE_PREFIX_PATH "/home/linuxbrew/install/EzPC/SCI/build/install")
add_definitions(-DSCI_HE)

find_package(GMP REQUIRED)
find_package(SCI REQUIRED)

include_directories("/home/linuxbrew/install/EzPC/SCI/build/install/include")

link_directories("/home/linuxbrew/install/EzPC/SCI/build/install/lib")

add_executable(boltmatmul main.cpp tensor.h module.h utils.h)
include_directories(boltmatmul ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx_sdk_v2 ${CMAKE_CURRENT_SOURCE_DIR}/../../c_sdk_v2)
target_link_libraries(boltmatmul PUBLIC cxx_sdk_v2 SCI::SCI-HE)
