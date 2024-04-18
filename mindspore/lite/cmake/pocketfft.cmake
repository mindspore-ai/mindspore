set(Pocketfft_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Pocketfft_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://github.com/malfet/pocketfft/archive/refs/heads/cpp.zip")
set(SHA256 "7c475524c264c450b78e221046d90b859316e105d3d6a69d5892baeafad95493")
set(INCLUDE "./")

mindspore_add_pkg(pocketfft
        HEAD_ONLY ./
        URL ${REQ_URL}
        SHA256 ${SHA256}
        )
include_directories(${pocketfft_INC})
