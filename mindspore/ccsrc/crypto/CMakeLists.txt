file(GLOB_RECURSE _CRYPTO_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_crypto_obj OBJECT ${_CRYPTO_SRC_FILES})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(_mindspore_crypto_obj mindspore::crypto)
endif()
