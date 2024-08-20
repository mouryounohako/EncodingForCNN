list(APPEND CMAKE_MODULE_PATH "/home/linuxbrew/install/EzPC/SCI/build/install/lib/cmake/SCI")
set(CMAKE_PREFIX_PATH "/home/linuxbrew/install/EzPC/SCI/build/install")
set(HEXL_HINT_DIR "/home/linuxbrew/install/hexl/installation/lib/cmake/hexl-1.2.5")

# add_definitions(-DSCI_OT)
add_definitions(-DSCI_HE)
find_package(GMP REQUIRED)
find_package(SCI REQUIRED)
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

add_executable(resplain main.cpp tensor.h module.h utils.h)
include_directories(resplain ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx_sdk_v2 ${CMAKE_CURRENT_SOURCE_DIR}/../../c_sdk_v2)
target_compile_definitions(resplain PUBLIC MULT_THREAD_LINEAR)
# MULT_THREAD_LINEAR
target_link_libraries(resplain PUBLIC cxx_sdk_v2 SCI::SCI-HE HEXL::hexl)
# target_link_libraries(resnet18fpga PUBLIC cxx_sdk_v2 SCI::SCI-HE)

