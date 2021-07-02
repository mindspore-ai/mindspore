file(GLOB_RECURSE ARMOUR_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set(SERVER_FLATBUFFER_OUTPUT "${CMAKE_BINARY_DIR}/schema")
set(FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../../schema/cipher.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/../../schema/fl_job.fbs
        )

set_property(SOURCE ${ARMOUR_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ARMOUR)
add_library(_mindspore_armour_obj OBJECT ${ARMOUR_FILES})
add_dependencies(_mindspore_armour_obj generated_fbs_files)
target_link_libraries(_mindspore_armour_obj mindspore::flatbuffers)