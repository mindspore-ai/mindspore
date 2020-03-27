SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -fPIC -O0 -Wall -Wno-deprecated-declarations -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -D_LIBCPP_INLINE_VISIBILITY='' -D'_LIBCPP_EXTERN_TEMPLATE(...)='")
SET(CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -fPIC -O3 -Wall -Wno-deprecated-declarations")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#add flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/include -Werror")


include_directories(./include)
add_subdirectory(src)
