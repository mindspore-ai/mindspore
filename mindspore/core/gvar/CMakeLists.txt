file(GLOB_RECURSE MS_GVAR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
set_property(SOURCE ${MS_GVAR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -O2")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
    add_compile_definitions(BUILDING_DLL)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wuser-defined-warnings -Winconsistent-missing-override")
endif()

add_library(mindspore_gvar SHARED ${MS_GVAR_SRC_LIST})

if(APPLE)
    set_target_properties(mindspore_gvar PROPERTIES MACOSX_RPATH ON)
endif()