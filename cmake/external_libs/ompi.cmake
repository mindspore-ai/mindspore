
set(ompi_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(ompi
        VER 3.1.5
        LIBS mpi
        URL https://github.com/open-mpi/ompi/archive/v3.1.5.tar.gz
        MD5 f7f220b26532c11a2efbc0bb73af3282
        PRE_CONFIGURE_COMMAND ./autogen.pl
        CONFIGURE_COMMAND ./configure)
include_directories(${ompi_INC})
add_library(mindspore::ompi ALIAS ompi::mpi)