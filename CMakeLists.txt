project(CameraDirectLinearTransformation)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0 FATAL_ERROR)

set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/")
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/" CACHE PATH "Single directory for all libraries")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/" CACHE PATH "Single directory for all executables")


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")

##########  CROSS-PLATFORM OPENGL + GLUT + GLEW  ########################
############################ FIND OPENGL EXTENSIONS GLEW ################

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_BINARY_DIR} )

add_executable(testCameraCalibration testCameraCalibration.cpp CameraDirectLinearTransformation.cpp)

