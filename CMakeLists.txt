CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

PROJECT(PiVA)

SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

FIND_PACKAGE(OpenCV REQUIRED)
find_package(OpenMP)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lpthread ${OpenMP_CXX_FLAGS}")


LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/3rdparty/ncnn/lib)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
	            ${PROJECT_SOURCE_DIR}/3rdparty/ncnn/include
		    ${PROJECT_SOURCE_DIR}/src)

############################

ADD_EXECUTABLE(piva 
               src/piva.cpp
               src/priorbox.cpp
               src/mobilefacenet.cpp
               src/live.cpp
               src/facedetection.cpp
               src/facedatabase.cpp)
TARGET_LINK_LIBRARIES(piva ncnn ${OpenCV_LIBS})
INSTALL(TARGETS piva
        DESTINATION ${CMAKE_INSTALL_PREFIX})

############################

ADD_EXECUTABLE(cropface
               src/cropface.cpp
               src/priorbox.cpp
               src/facedetection.cpp)
TARGET_LINK_LIBRARIES(cropface ncnn ${OpenCV_LIBS})
INSTALL(TARGETS cropface
        DESTINATION ${CMAKE_INSTALL_PREFIX})

############################

ADD_EXECUTABLE(control src/cameracontrol.cpp)
INSTALL(TARGETS control
        DESTINATION ${CMAKE_INSTALL_PREFIX})

############################

ADD_EXECUTABLE(makefrdb
               src/makefrdb.cpp
               src/mobilefacenet.cpp
               src/facedatabase.cpp)
TARGET_LINK_LIBRARIES(makefrdb ncnn ${OpenCV_LIBS})
INSTALL(TARGETS makefrdb
        DESTINATION ${CMAKE_INSTALL_PREFIX})

