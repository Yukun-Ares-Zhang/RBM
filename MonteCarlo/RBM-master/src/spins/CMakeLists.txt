cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(alpsize NONE)

# find ALPS Library
find_package(ALPS REQUIRED PATHS ${ALPS_ROOT_DIR} $ENV{ALPS_HOME} NO_SYSTEM_ENVIRONMENT_PATH)
message(STATUS "Found ALPS: ${ALPS_ROOT_DIR} (revision: ${ALPS_VERSION})")
include(${ALPS_USE_FILE})

# enable C and C++ compilers
enable_language(C CXX)

# rule for generating wolff program
add_executable(ising ising.C)
target_link_libraries(ising ${ALPS_LIBRARIES})

#add_executable(cop_ising cop_ising.C)
#target_link_libraries(cop_ising ${ALPS_LIBRARIES})

#add_executable(potts potts.C)
#target_link_libraries(potts ${ALPS_LIBRARIES})

#add_executable(xy xy.C)
#target_link_libraries(xy ${ALPS_LIBRARIES})
