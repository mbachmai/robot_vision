# CMakeLists.txt

# Older versions of CMake are likely to work just fine but, since
# I don't know where to cut off I just use the version I'm using
cmake_minimum_required(VERSION "3.10")

# name of this example project
project(assign1)

# set OpenCV_DIR variable equal to the path to the cmake
# files within the previously installed opencv program
set(OpenCV_DIR /usr/local/include/opencv4)

# Tell compiler to use C++ 14 features which is needed because
# Clang version is often behind in the XCode installation
set(CMAKE_CXX_STANDARD 20)

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package( OpenCV REQUIRED )

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )

# specify the executable target to be built
add_executable(code assign1_task1.cpp)
add_executable(stereocalib assign1_task2.cpp)
add_executable(stereomatch assign1_task3.cpp)
add_executable(featurematch assign2_task1.cpp)
add_executable(fundmatrix assign2_task2.cpp)

# tell it to link the executable target against OpenCV
target_link_libraries(code ${OpenCV_LIBS} )
target_link_libraries(stereocalib ${OpenCV_LIBS} )
target_link_libraries(stereomatch ${OpenCV_LIBS} )
target_link_libraries(featurematch ${OpenCV_LIBS} )
target_link_libraries(fundmatrix ${OpenCV_LIBS} )
