# Note: OpenCL-CLHPP depends on OpenCL-Headers
if(ENABLE_GITEE_EULER)
    # Already downloaded in opencl-header.cmake
elseif(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/OpenCL-CLHPP/repository/archive/v2.0.12.tar.gz")# VER 2.0.12
    set(SHA256 "d5bdbfb614a6494de97abf7297db6d2c88a55a095b12949d797ce562f5d4fdce")
    __download_pkg(OpenCL-CLHPP ${REQ_URL} ${SHA256})
else()
    set(REQ_URL "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v2.0.12.tar.gz")
    set(SHA256 "20b28709ce74d3602f1a946d78a2024c1f6b0ef51358b9686612669897a58719")
    __download_pkg(OpenCL-CLHPP ${REQ_URL} ${SHA256})
endif()