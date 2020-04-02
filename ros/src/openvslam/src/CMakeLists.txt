find_package(catkin REQUIRED COMPONENTS tf)

catkin_package(
    CATKIN_DEPENDS tf
)

if(USE_PANGOLIN_VIEWER)
    message(STATUS "Viewer for examples: pangolin viewer")
elseif(USE_SOCKET_PUBLISHER)
    message(STATUS "Viewer for examples: socket publisher")
else()
    message(STATUS "Viewer for examples: none")
endif()

set(EXECUTABLE_TARGETS "")

#add_library(${PROJECT_NAME}
#            SHARED
#            ${CMAKE_CURRENT_SOURCE_DIR}/MsgSync/MsgSynchronizer.h
#            ${CMAKE_CURRENT_SOURCE_DIR}/MsgSync/MsgSynchronizer.cpp)


add_executable(run_slam run_slam.cc)
list(APPEND EXECUTABLE_TARGETS run_slam)

add_executable(run_localization run_localization.cc)
list(APPEND EXECUTABLE_TARGETS run_localization)

add_executable(rgbd_slam rgbd_slam.cc)
list(APPEND EXECUTABLE_TARGETS rgbd_slam)

add_executable(stereo_slam stereo_slam.cc)
list(APPEND EXECUTABLE_TARGETS stereo_slam)

add_executable(mono_slam mono_slam.cc)
list(APPEND EXECUTABLE_TARGETS mono_slam)

add_executable(IMU_slam IMU_slam.cc)
list(APPEND EXECUTABLE_TARGETS IMU_slam)

foreach(EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    add_dependencies(${EXECUTABLE_TARGET} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

    # pangolin_viewer is used on a priority basis
    if(USE_PANGOLIN_VIEWER)
        # set macro flag
        target_compile_definitions(${EXECUTABLE_TARGET}
                PRIVATE -DUSE_PANGOLIN_VIEWER)
        # link viewer
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE
                ${Pangolin_Viewer_LIB} ${Pangolin_LIBRARIES} ${catkin_LIBRARIES})
    elseif(USE_SOCKET_PUBLISHER)
        # set macro flag
        target_compile_definitions(${EXECUTABLE_TARGET}
                PRIVATE -DUSE_SOCKET_PUBLISHER)
        # link viewer
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE
                ${Socket_Publisher_LIB} ${SIOCLIENT_LIBRARY} ${PROTOBUF_LIBRARIES})
    endif()

    # setup stack trace logger
    if(USE_STACK_TRACE_LOGGER)
        target_compile_definitions(${EXECUTABLE_TARGET}
                PRIVATE -DUSE_STACK_TRACE_LOGGER)
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE
                ${GLOG_LIBRARIES})
    endif()

    # setup google-perftools
    if(USE_GOOGLE_PERFTOOLS)
        target_compile_definitions(${EXECUTABLE_TARGET}
                PRIVATE -DUSE_GOOGLE_PERFTOOLS)
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE
                ${GPERFTOOLS_LIBRARIES})
    endif()

endforeach()
