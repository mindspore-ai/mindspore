#add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

if (ENABLE_ACL)
    add_subdirectory(cxx_st)
else ()
    add_subdirectory(ut)
endif ()
