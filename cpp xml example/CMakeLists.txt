cmake_minimum_required(VERSION 3.10)
project(MuJoCoGLFW)

set(CMAKE_CXX_STANDARD 17)

# Set the paths to your dependencies
set(MUJOCO_PATH /Applications/mujoco.framework)
set(GLFW_PATH /opt/homebrew/opt/glfw)
set(EIGEN_INCLUDE_DIR /opt/homebrew/opt/eigen/include/eigen3)
set(TINYXML2_LIB_DIR /opt/homebrew/lib)
set(TINYXML2_INCLUDE_DIR /opt/homebrew/include)

include_directories(${MUJOCO_PATH}/Headers)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${TINYXML2_INCLUDE_DIR})
include_directories(${GLFW_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/include)

link_directories(${GLFW_PATH}/lib)
link_directories(${TINYXML2_LIB_DIR})
link_directories(/opt/homebrew/lib)

add_executable(MuJoCoGLFW
    src/main.cpp
    src/GLFWWindowManager.cpp
)

target_link_libraries(MuJoCoGLFW
    glfw
    tinyxml2
    dl
    pthread
)

# Ensure MuJoCo framework is linked correctly
target_link_options(MuJoCoGLFW PRIVATE
    -F/Applications
    -framework mujoco
    -Wl,-rpath,@executable_path/../Frameworks
    -Wl,-rpath,/Applications/mujoco.framework/Versions/A
)

# Set the sysroot explicitly if not already set
if(NOT DEFINED CMAKE_OSX_SYSROOT)
    set(CMAKE_OSX_SYSROOT ${SDKROOT})
endif()
