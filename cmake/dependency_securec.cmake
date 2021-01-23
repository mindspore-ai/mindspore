# securec library
#
#
# SECUREC_LIBRARY
#

if(NOT TARGET securec)
  set(_ms_tmp_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(_ms_tmp_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

  set(CMAKE_C_FLAGS "${SECURE_CXX_FLAGS}")
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_compile_definitions(SECUREC_ONLY_DECLARE_MEMSET)
  endif()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/securec ${CMAKE_BINARY_DIR}/securec)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${_ms_tmp_CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_C_FLAGS ${_ms_tmp_CMAKE_C_FLAGS})
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/securec/include)

set(SECUREC_LIBRARY securec)
