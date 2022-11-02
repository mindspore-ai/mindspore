if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ompi/repository/archive/v4.0.3.tar.gz")
    set(SHA256 "2cc45973dce8f147f747a8f2c4c46d171f7f3a142548812a5649e78b84a55f60")
else()
    set(REQ_URL "https://github.com/open-mpi/ompi/archive/v4.0.3.tar.gz")
    set(SHA256 "5663679c17c26cf8dfd5fe1eea79c2d7476f408cc22933b9750af2158ec1657b")
endif()

set(ompi_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(ompi
        VER 4.0.3
        LIBS mpi
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PRE_CONFIGURE_COMMAND ./autogen.pl
        CONFIGURE_COMMAND ./configure --disable-mpi-fortran)
include_directories(${ompi_INC})
add_library(mindspore::ompi ALIAS ompi::mpi)