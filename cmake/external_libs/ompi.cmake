if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ompi/repository/archive/v4.0.3.tar.gz")
    set(MD5 "f76abc92ae870feff186d790f40ae762")
else()
    set(REQ_URL "https://github.com/open-mpi/ompi/archive/v4.0.3.tar.gz")
    set(MD5 "86cb724e8fe71741ad3be4e7927928a2")
endif()

set(ompi_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(ompi
        VER 4.0.3
        LIBS mpi
        URL ${REQ_URL}
        MD5 ${MD5}
        PRE_CONFIGURE_COMMAND ./autogen.pl
        CONFIGURE_COMMAND ./configure)
include_directories(${ompi_INC})
add_library(mindspore::ompi ALIAS ompi::mpi)