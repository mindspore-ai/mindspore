#add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

if (ENABLE_ACL)
    add_subdirectory(cxx_st)
elseif (ENABLE_GPU OR ENABLE_D OR ENABLE_CPU)
    message(fatal "No need set -e xxx when compile ut")
else ()
    add_subdirectory(ut)
endif()
