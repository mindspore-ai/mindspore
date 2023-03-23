#add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

if(ENABLE_TESTCASES)
    add_subdirectory(ut)
elseif(ENABLE_CPP_ST)
    add_subdirectory(st/cpp/cxx_api)
    add_subdirectory(st/cpp/c_api)
endif()
