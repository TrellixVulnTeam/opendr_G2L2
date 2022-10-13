cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

set(OPENDR_LIB)
set(OPENDR_INC)

set(OpenCV_LIB_DIR "/usr/lib/x86_64-linux-gnu")
LINK_DIRECTORIES(
    ${OPENDR_LIB}
    ${OpenCV_LIB_DIR}
)

project(nanodet_demo)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${OPENDR_INC}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")

add_executable(nanodet_demo nanodet_demo.c)


target_link_libraries(
    nanodet_demo
    opendr
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "OPENDR_LIBRARIES = nanodet_demo")
message(STATUS "OPENDR_LIBRARIES = lopendr")
message(STATUS "TORCH_CXX_FLAGS = ${TORCH_CXX_FLAGS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
