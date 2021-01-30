# include DIRECTIVES FOR LITEBUS LIBRARY (generates, e.g., -I/path/to/thing on Linux).
######################################################################################
if(${HTTP_ENABLED} STREQUAL "on")
  set(LITEBUS_INCLUDE_DIR ${LITEBUS_INCLUDE_DIR} ${HTTP_PARSER_INCLUDE_DIR})
endif()

if(${SSL_ENABLED} STREQUAL "on")
  set(LITEBUS_INCLUDE_DIR ${LITEBUS_INCLUDE_DIR}
    ${LITEBUS_OSSCRYPTO_INCLUDE_DIR}
    ${LITEBUS_HARESCRYPTO_INCLUDE_DIR}
  )
endif()

link_directories(${SECUREC_LIB_DIR})
link_directories(${GLOG_LIB_DIR})
link_directories(${HARES_LOG_LIB_DIR})
link_directories(${PROTOBUF_C_LIB_DIR})

if(${HTTP_ENABLED} STREQUAL "on")
  link_directories(${HTTP_PARSER_LIB_DIR})
endif()

# LINK libgcov.a
#######################################################################
if(${CODE_COVERAGE} STREQUAL "on")
    set(LINK_LIBS ${LINK_LIBS} gcov)
endif()

# add object lib to avoid duplicate compile for a single source file
#######################################################################
add_library(litebus_obj OBJECT litebus.cc)
target_include_directories(litebus_obj PUBLIC ${LITEBUS_INCLUDE_DIR} ${3RDPARTY_LITEBUS_INCLUDE_DIRS})
#add_library(decrypt_obj OBJECT)

# THE LITEBUS LIBRARY (generates, e.g., liblitebus.so, etc., on Linux).
#######################################################################
if(${STATIC_LIB} STREQUAL "on")

#######################################################################
add_library(${LITEBUS_TARGET}_static STATIC $<TARGET_OBJECTS:litebus_obj>)
target_link_libraries(${LITEBUS_TARGET}_static ${LINK_LIBS} ${LITEBUS_HARES_DECRYPT_SLIB}
${OPENSSL_SSL_LIB_A} ${OPENSSL_CRYPTO_LIB_A})
set_target_properties(${LITEBUS_TARGET}_static PROPERTIES OUTPUT_NAME ${LITEBUS_TARGET})

if(DEFINED DEPEND_PATH)
  add_custom_command(TARGET ${LITEBUS_TARGET}_static POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink lib${LITEBUS_TARGET}.a liblitebus.a
  )
endif()

#######################################################################
endif()

  set(LINK_LIBS ${LINK_LIBS})

if(${HTTP_ENABLED} STREQUAL "on")
  set(LINK_LIBS ${LINK_LIBS} ${HTTP_PARSER_DFLAG})
endif()

add_library(litebus_shared SHARED $<TARGET_OBJECTS:litebus_obj>)
target_link_libraries(litebus_shared ${LINK_LIBS})
target_include_directories(litebus_shared PUBLIC ${LITEBUS_INCLUDE_DIR} ${3RDPARTY_LITEBUS_INCLUDE_DIRS})

set_target_properties(
    litebus_shared PROPERTIES
    OUTPUT_NAME litebus
    VERSION ${LITEBUS_PACKAGE_VERSION}
    SOVERSION ${LITEBUS_PACKAGE_VERSION}
    LINK_FLAGS -s
)

#copy lib to depend path (internal use)
#set(DEPEND_PATH "${PROJECT_SOURCE_DIR}/output1")
if(DEFINED DEPEND_PATH)
  set(DEPEND_LIB_PATH ${DEPEND_PATH}/LITEBUS/lib)
  set(DEPEND_INCLUDE_PATH ${DEPEND_PATH}/LITEBUS/include)

  add_custom_target(litebus_all ALL COMMENT "================= litebus_all =====================")
  if(${STATIC_LIB} STREQUAL "on")
    add_dependencies(litebus_all litebus_shared ${LITEBUS_TARGET}_static)
  endif()
  add_dependencies(litebus_all litebus_shared)
  add_custom_command(TARGET litebus_all POST_BUILD
  COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/MakeDirectory.cmake
          ${DEPEND_LIB_PATH} ${DEPEND_INCLUDE_PATH}
  COMMAND ${CMAKE_COMMAND} -DLITEBUS_COPYTO="${DEPEND_LIB_PATH}" -P
          ${PROJECT_SOURCE_DIR}/cmake/CopyLibToPath.cmake
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/litebus.hpp
          ${DEPEND_INCLUDE_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/litebus.h
          ${DEPEND_INCLUDE_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/actor
          ${DEPEND_INCLUDE_PATH}/actor
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/async
          ${DEPEND_INCLUDE_PATH}/async
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/timer
          ${DEPEND_INCLUDE_PATH}/timer
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/exec
          ${DEPEND_INCLUDE_PATH}/exec
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/utils
          ${DEPEND_INCLUDE_PATH}/utils)
  if(${HTTP_ENABLED} STREQUAL "on")
    add_custom_command(TARGET litebus_all POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_SOURCE_DIR}/include/httpd ${DEPEND_INCLUDE_PATH}/httpd)
  endif()
  if(${SSL_ENABLED} STREQUAL "on")
    add_custom_command(TARGET litebus_all POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_SOURCE_DIR}/include/ssl ${DEPEND_INCLUDE_PATH}/ssl)
  endif()

endif()

#install lib to package path
if("${PROJECT_HARES}" STREQUAL "cloudcore")
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_LS_Master/lib PERMISSIONS OWNER_READ)
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_LS_Slave/lib PERMISSIONS OWNER_READ)
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_HASEN_Common/lib PERMISSIONS OWNER_READ)
elseif("${PROJECT_HARES}" STREQUAL "hasen_cloudcore_csp")
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_Common/lib PERMISSIONS OWNER_READ)
else()
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_Common/lib PERMISSIONS OWNER_READ)
    install(TARGETS litebus_shared LIBRARY DESTINATION HARES_Slave/lib PERMISSIONS OWNER_READ)
endif()

# Build pbjson.so.
add_subdirectory(actor)
add_subdirectory(async)
add_subdirectory(evloop)
add_subdirectory(timer)