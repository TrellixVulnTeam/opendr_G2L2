cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_LIB_DIR "/usr/lib/x86_64-linux-gnu")
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

project(nanodet_demo)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")

add_executable(nanodet_demo main.cpp nanodet_libtorch.cpp opendr_utils.cpp)



target_link_libraries(
    nanodet_demo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET nanodet_demo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:nanodet_demo>)
endif (MSVC)
