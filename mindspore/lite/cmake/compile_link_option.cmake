
if(MSVC)
    add_compile_definitions(SUPPORT_MSVC)
    add_compile_definitions(_ENABLE_ATOMIC_ALIGNMENT_FIX)
    set(CMAKE_C_FLAGS "/O2 /EHsc /GS /Zi /utf-8")
    set(CMAKE_CXX_FLAGS "/O2 /EHsc /GS /Zi /utf-8 /std:c++17")
    set(CMAKE_SHARED_LINKER_FLAGS "/DEBUG ${SECURE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "/DEBUG ${SECURE_SHARED_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
else()
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations \
        -Wno-missing-braces  ${SECURE_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations \
        -Wno-missing-braces -Wno-overloaded-virtual -std=c++17 ${SECURE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

    set(CMAKE_C_FLAGS_DEBUG "-DDebug -g -fvisibility=default")
    set(CMAKE_CXX_FLAGS_DEBUG "-DDebug -g -fvisibility=default")

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        string(REPLACE "-O2" "-O0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE "-O2" "-O0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
    set(CMAKE_SHARED_LINKER_FLAGS "${SECURE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${SECURE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
endif()
