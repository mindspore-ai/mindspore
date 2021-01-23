SET(CMAKE_BUILD_TYPE "Debug")
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    SET(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -fPIC -O0 -Wall -Wno-deprecated-declarations -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -fstack-protector-all")
else ()
    SET(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -Wno-nullability-completeness -fPIC -O0 -Wall -Wno-deprecated-declarations -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -fstack-protector-all -D_LIBCPP_INLINE_VISIBILITY='' -D'_LIBCPP_EXTERN_TEMPLATE(...)='")
endif()
SET(CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -fPIC -O3 -Wall -Wno-deprecated-declarations -fstack-protector-all")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#add flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/include -Werror")

include_directories(./include)
add_subdirectory(src)
