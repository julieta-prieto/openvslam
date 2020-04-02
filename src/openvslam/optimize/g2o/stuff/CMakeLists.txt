# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/string_tools.h
               ${CMAKE_CURRENT_SOURCE_DIR}/color_macros.h 
               ${CMAKE_CURRENT_SOURCE_DIR}/macros.h
               ${CMAKE_CURRENT_SOURCE_DIR}/timeutil.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/misc.h
               ${CMAKE_CURRENT_SOURCE_DIR}/timeutil.h
               ${CMAKE_CURRENT_SOURCE_DIR}/os_specific.c    
               ${CMAKE_CURRENT_SOURCE_DIR}/os_specific.h
               ${CMAKE_CURRENT_SOURCE_DIR}/string_tools.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/property.cpp       
               ${CMAKE_CURRENT_SOURCE_DIR}/property.h
               ${CMAKE_CURRENT_SOURCE_DIR}/sparse_helper.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/sparse_helper.h)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/stuff)
