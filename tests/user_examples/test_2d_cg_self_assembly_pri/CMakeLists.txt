set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SPHINXSYS_PROJECT_DIR}/cmake)

set(CMAKE_VERBOSE_MAKEFILE on)

# Keep the build target short enough for MSVC's generated object/PDB paths.
project("test_2d_cg_assembly_pri")

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

# Bake the source revision into the executable so every run can report the
# commit it was actually built from, instead of relying on someone typing it.
#
# `describe --always` always yields something (falls back to the abbreviated
# commit hash when no tag is reachable), and `--dirty` appends "-dirty" when
# the worktree has uncommitted modifications of tracked files.  That is what
# lets two machines tell "same commit, clean tree" apart from "same commit
# plus local edits", which matters as soon as they run case comparisons
# against each other.
#
# LIMITATION: this runs at CONFIGURE time, so the baked string goes stale if
# sources are edited and only the build step is re-run.  It is a lower bound
# on cleanliness, not a guarantee -- the launcher (run_cg.ps1) records
# `git describe --always --dirty` at launch time and passes it through
# --git-commit, which overrides the baked value.  Keep the two together.
execute_process(COMMAND git describe --always --dirty --abbrev=7
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE CG_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT CG_GIT_COMMIT)
    set(CG_GIT_COMMIT "unknown")
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE CG_GIT_COMMIT="${CG_GIT_COMMIT}")

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
target_link_libraries(${PROJECT_NAME} sphinxsys_2d)
