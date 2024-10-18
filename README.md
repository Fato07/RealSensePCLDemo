# Realsense PCL Example

This project demonstrates how to use the Intel RealSense cameras and the Point Cloud Library (PCL) for plane fitting and segmentation. The project builds using **CMake** and supports both **Windows** (with vcpkg) and **macOS** (with Homebrew).

## Prerequisites

### Windows:
- CMake 3.0 or above
- vcpkg for dependency management

### macOS:
- CMake 3.0 or above
- Homebrew for dependency management

## Dependencies

The following dependencies are required:
- Boost
- PCL (Point Cloud Library)
- Intel RealSense SDK
- OpenCV
- flann
- lz4
- quirc

## Building the Project

### Windows:

1. Install vcpkg (if not already installed):
    ```bash
    git clone https://github.com/Microsoft/vcpkg.git
    ./vcpkg/bootstrap-vcpkg.bat
    ```
   
2. Install dependencies:
    ```bash
    vcpkg install boost flann lz4 pcl realsense2 opencv quirc
    ```

3. Configure and build:
    ```bash
    cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
    make
    ```

### macOS:

1. Install dependencies using Homebrew:
    ```bash
    brew install boost flann lz4 pcl librealsense opencv quirc
    ```

2. Configure and build:
    ```bash
    cmake ..
    make
    ```
make -I /Users/fathindosunmu/quirc

## Running the Program

After building, run the executable:
```bash
./realsense_pcl_example
