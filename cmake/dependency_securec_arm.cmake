# securec arm library
#
#
# SECUREC_ARM_LIBRARY
#

if(NOT TARGET securec_arm)
  set(_ms_tmp_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(_ms_tmp_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

  if(TARGET_OHOS_LITE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SECURE_CXX_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "${SECURE_CXX_FLAGS}")
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_compile_definitions(SECUREC_ONLY_DECLARE_MEMSET)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ${_ms_tmp_CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_C_FLAGS ${_ms_tmp_CMAKE_C_FLAGS})
endif()

if(NOT MSVC)
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -fPIC -O2 -Wall -Wno-deprecated-declarations \
            -fno-inline-functions -fno-omit-frame-pointer -fstack-protector-all")
    else()
        SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -Wno-nullability-completeness -fPIC -O2 -Wall \
            -Wno-deprecated-declarations -fno-inline-functions -fno-omit-frame-pointer \
            -fstack-protector-all -D_LIBCPP_INLINE_VISIBILITY='' -D'_LIBCPP_EXTERN_TEMPLATE(...)='")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    #add flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/include -Werror")
endif()

include_directories(${PROJECT_SOURCE_DIR}/third_party/securec/include)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND SECUREC_ARM_LIBRARY "memset_s.c")
else()
    aux_source_directory(${PROJECT_SOURCE_DIR}/third_party/securec/src SECUREC_ARM_LIBRARY)
endif()
add_library(securec_arm STATIC ${SECUREC_ARM_LIBRARY})
set(SECUREC_ARM_LIBRARY securec_arm)
