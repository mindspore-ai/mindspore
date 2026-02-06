set(Pocketfft_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Pocketfft_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/archive/release_for_eigen/pocketfft-release_for_eigen.zip")
set(SHA256 "d3b88763e7e069bab2041e952990c677acda83a19e85ea015eefb1816d7a3344")
set(INCLUDE "./")

mindspore_add_pkg(pocketfft
        HEAD_ONLY ./
        URL ${REQ_URL}
        SHA256 ${SHA256}
        )
include_directories(${pocketfft_INC})
