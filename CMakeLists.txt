set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(HOMEBREW_PREFIX "/usr/local")  # Use /opt/homebrew for Apple Silicon

set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
set(CMAKE_INCLUDE_PATH "${HOMEBREW_PREFIX}/include")
set(CMAKE_LIBRARY_PATH "${HOMEBREW_PREFIX}/lib")

cmake_minimum_required(VERSION 3.0)
project(realsense_pcl_example)

# Use the vcpkg toolchain to find packages automatically
# Ensure the CMake command is run with the correct vcpkg toolchain
# cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Users/Raido/Downloads/vcpkg/scripts/buildsystems/vcpkg.cmake

# Find required packages
find_package(Boost REQUIRED COMPONENTS system iostreams filesystem serialization)

find_package(flann CONFIG REQUIRED)

# Manually specify the paths for lz4
find_library(LZ4_LIBRARY lz4 HINTS /usr/local/lib /opt/homebrew/lib)
find_path(LZ4_INCLUDE_DIR lz4.h HINTS /usr/local/include /opt/homebrew/include)

find_package(PCL REQUIRED)
find_package(realsense2 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
#find_package(quirc CONFIG REQUIRED)


# Add include directories and link libraries manually
include_directories(${LZ4_INCLUDE_DIR})

# Include directories (vcpkg will handle paths automatically)
include_directories(${PCL_INCLUDE_DIRS} ${realsense2_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Add the executable
add_executable(realsense_pcl_example plane_fitter.cpp)

# Link the libraries (vcpkg will handle the linking automatically)
target_link_libraries(realsense_pcl_example ${PCL_LIBRARIES} realsense2::realsense2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
