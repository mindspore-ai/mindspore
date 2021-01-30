target_sources(litebus_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/async.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/future.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/uuid_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/uuid_generator.cc
#               ${CMAKE_CURRENT_SOURCE_DIR}/flag_parser_impl.cpp
)
