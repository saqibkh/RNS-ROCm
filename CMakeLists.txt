cmake_minimum_required(VERSION 3.21)
project(rocm_rns_matmul LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(RNS_ENABLE_GPU "Enable GPU (ROCm/HIP) support" ON)
option(RNS_ENABLE_TESTS "Enable unit tests" ON)
option(RNS_ENABLE_EXAMPLES "Enable example programs" ON)

# Find Boost for multiprecision (header-only)
find_package(Boost QUIET)
if(Boost_FOUND)
  message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}")
  add_definitions(-DRNS_HAS_BOOST)
else()
  message(STATUS "Boost not found - using internal bigint implementation")
endif()

# GPU support via HIP
if(RNS_ENABLE_GPU)
  # Try to find HIP
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/hip")
  find_package(hip QUIET)
  
  if(hip_FOUND)
    message(STATUS "Found HIP: ${hip_VERSION}")
    enable_language(HIP)
    set(RNS_HAS_GPU TRUE)
    add_definitions(-DRNS_HAS_GPU)
  else()
    message(WARNING "HIP not found - building CPU-only version")
    set(RNS_HAS_GPU FALSE)
    set(RNS_ENABLE_GPU OFF)
  endif()
else()
  set(RNS_HAS_GPU FALSE)
endif()

# Source files
set(RNS_SOURCES
  src/rns_api.cpp
  src/crt.cpp
  src/utils.cpp
  src/primes.cpp
)

if(RNS_HAS_GPU)
  list(APPEND RNS_SOURCES
    src/hip/rns_kernels.hip
    src/hip/rns_matmul_m4.hip
    src/hip/rns_matmul_m6.hip
    src/hip/rns_matmul_m8.hip
    src/hip/rns_matmul_m10.hip
  )
endif()

# Main library
add_library(rns_rocm ${RNS_SOURCES})

target_include_directories(rns_rocm PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(Boost_FOUND)
  target_include_directories(rns_rocm PUBLIC ${Boost_INCLUDE_DIRS})
endif()

if(RNS_HAS_GPU)
  set_target_properties(rns_rocm PROPERTIES
    HIP_ARCHITECTURES "gfx90a;gfx908;gfx1030"
  )
  target_link_libraries(rns_rocm PUBLIC hip::device)
endif()

# Tests
if(RNS_ENABLE_TESTS)
  enable_testing()
  
  # CPU tests (always build)
  add_executable(test_rns_ops tests/test_rns_ops.cpp)
  target_link_libraries(test_rns_ops PRIVATE rns_rocm)
  add_test(NAME test_rns_ops COMMAND test_rns_ops)
  
  add_executable(test_crt tests/test_crt.cpp)
  target_link_libraries(test_crt PRIVATE rns_rocm)
  add_test(NAME test_crt COMMAND test_crt)
  
  add_executable(test_crt_matrix tests/test_crt_matrix.cpp)
  target_link_libraries(test_crt_matrix PRIVATE rns_rocm)
  add_test(NAME test_crt_matrix COMMAND test_crt_matrix)
  
  if(RNS_HAS_GPU)
    add_executable(test_gemm_mod tests/test_gemm_mod.cpp)
    target_link_libraries(test_gemm_mod PRIVATE rns_rocm)
    add_test(NAME test_gemm_mod COMMAND test_gemm_mod)
  endif()
endif()

# Examples
if(RNS_ENABLE_EXAMPLES)
  if(RNS_HAS_GPU)
    add_executable(demo_gemm examples/demo_gemm.cpp)
    target_link_libraries(demo_gemm PRIVATE rns_rocm)
  endif()
  
  add_executable(demo_crt_matrix examples/demo_crt_matrix.cpp)
  target_link_libraries(demo_crt_matrix PRIVATE rns_rocm)
endif()

# Install
install(TARGETS rns_rocm
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/rns DESTINATION include)

# Summary
message(STATUS "")
message(STATUS "=== RNS-RoCM Configuration ===")
message(STATUS "  GPU support:    ${RNS_HAS_GPU}")
message(STATUS "  Tests:          ${RNS_ENABLE_TESTS}")
message(STATUS "  Examples:       ${RNS_ENABLE_EXAMPLES}")
message(STATUS "  Boost bigint:   ${Boost_FOUND}")
message(STATUS "")
