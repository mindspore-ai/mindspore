#add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

if(ENABLE_TESTCASES)
    add_subdirectory(ut)
else()
    if(ENABLE_CPP_ST)
        add_subdirectory(st/cpp/cxx_api)
    endif()
    if(ENABLE_C_ST)
        add_subdirectory(st/cpp/c_api)
    endif()
endif()
