# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE

               ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_cholmod.h
               ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_eigen.h
               ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver_dense.h)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/solvers)
