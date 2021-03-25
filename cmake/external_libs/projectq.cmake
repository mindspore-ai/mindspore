set(projectq_CXXFLAGS "-fopenmp -O2 -ffast-mast -mavx -DINTRIN")
set(projectq_CFLAGS "-fopenmp -O2 -ffast-mast -mavx -DINTRIN")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ProjectQ/repository/archive/v0.5.1.tar.gz")
    set(MD5 "d874e93e56d3375f1c54c7dd1b731054")
else()
    set(REQ_URL "https://github.com/ProjectQ-Framework/ProjectQ/archive/v0.5.1.tar.gz ")
    set(MD5 "13430199c253284df8b3d840f11d3560")
endif()

if(ENABLE_CPU AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux"
    AND ${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    message("Include projectq simulator")
    mindspore_add_pkg(projectq
        VER 0.5.1
        HEAD_ONLY ./
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/projectq/projectq.patch001
    )
    include_directories(${projectq_INC})
else()
    message("Quantum simulation only support x86_64 linux platform.")
endif()
