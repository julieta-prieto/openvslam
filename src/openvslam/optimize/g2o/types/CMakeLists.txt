# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/types_sba.h
               ${CMAKE_CURRENT_SOURCE_DIR}/types_six_dof_expmap.h
               ${CMAKE_CURRENT_SOURCE_DIR}/types_sba.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/types_six_dof_expmap.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/types_seven_dof_expmap.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/types_seven_dof_expmap.h
               ${CMAKE_CURRENT_SOURCE_DIR}/se3quat.h
               ${CMAKE_CURRENT_SOURCE_DIR}/se3_ops.h
               ${CMAKE_CURRENT_SOURCE_DIR}/se3_ops.hpp)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/types)
