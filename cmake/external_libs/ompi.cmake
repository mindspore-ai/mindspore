if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ompi/repository/archive/v4.1.4.tar.gz")
    set(SHA256 "b7c084ee3c292aba6caf02493dbeaf0767d0cb5b3a64289a0f74cb144bfaf230")
    set(PRE_CONFIGURE_CMD "./autogen.pl")
else()
    set(REQ_URL "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz")
    set(SHA256 "e166dbe876e13a50c2882e11193fecbc4362e89e6e7b6deeb69bf095c0f4fc4c")
    set(PRE_CONFIGURE_CMD "")
endif()

set(ompi_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(ompi
        VER 4.1.4
        LIBS mpi
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PRE_CONFIGURE_COMMAND ${PRE_CONFIGURE_CMD}
        CONFIGURE_COMMAND ./configure --disable-mpi-fortran)
include_directories(${ompi_INC})
add_library(mindspore::ompi ALIAS ompi::mpi)