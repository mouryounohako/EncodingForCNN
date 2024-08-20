list(APPEND CMAKE_MODULE_PATH "/home/linuxbrew/install/EzPC/SCI/build/install/lib/cmake/SCI")
set(CMAKE_PREFIX_PATH "/home/linuxbrew/install/EzPC/SCI/build/install")
set(HEXL_HINT_DIR "/home/linuxbrew/install/hexl/install/lib/cmake/hexl-1.2.5")

add_definitions(-DSCI_HE)

find_package(GMP REQUIRED)
find_package(SCI REQUIRED)

# Example using source
find_package(HEXL 1.2.5
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
if (NOT TARGET HEXL::hexl)
    message(FATAL_ERROR "TARGET HEXL::hexl not found")
endif()


include_directories("/home/linuxbrew/install/EzPC/SCI/build/install/include")
include_directories("/home/linuxbrew/install/hexl/installation/include")

link_directories("/home/linuxbrew/install/EzPC/SCI/build/install/lib")
link_directories("/home/linuxbrew/install/hexl/installation/lib")

add_executable(conv2dnest main.cpp tensor.h module.h utils.h)
include_directories(conv2dnest ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx_sdk_v2 ${CMAKE_CURRENT_SOURCE_DIR}/../../c_sdk_v2)
target_link_libraries(conv2dnest PUBLIC cxx_sdk_v2 SCI::SCI-HE HEXL::hexl)
