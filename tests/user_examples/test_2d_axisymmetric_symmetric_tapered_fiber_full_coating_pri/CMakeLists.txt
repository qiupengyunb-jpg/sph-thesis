set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SPHINXSYS_PROJECT_DIR}/cmake)

set(CMAKE_VERBOSE_MAKEFILE on)

# Keep the build target short enough for MSVC's generated object/PDB paths.
project("test_2d_axisym_fcf_film_pri")

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
target_link_libraries(${PROJECT_NAME} sphinxsys_2d)
