if(ENABLE_GITEE)
    set(REQ_URL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz")
    set(SHA256 "5692cc80554a7117c99eaa725d35100edd8bbf73423a5e265ff867979192df7d")
    set(PRE_CONFIGURE_CMD "")
else()
    set(REQ_URL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz")
    set(SHA256 "5692cc80554a7117c99eaa725d35100edd8bbf73423a5e265ff867979192df7d")
    set(PRE_CONFIGURE_CMD "")
endif()

set(ompi_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(ompi
        VER 5.0.10
        LIBS mpi
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PRE_CONFIGURE_COMMAND ${PRE_CONFIGURE_CMD}
        CONFIGURE_COMMAND ./configure --disable-mpi-fortran)

# OMPI 5.x's bundled libevent installs its headers into ${ompi_DIRPATH}/include.
# That copy is built with --disable-openssl, so its event-config.h has no
# EVENT__HAVE_OPENSSL define and bufferevent_ssl.h hides the openssl APIs.
# Because -I${ompi_DIRPATH}/include is searched before the real libevent's
# -isystem path, mindspore_cluster's tcp client/server fail to compile against
# libevent_openssl. The bundled libevent is only needed for OMPI's own runtime
# (statically baked into libprrte/libpmix), so strip these headers from the
# install dir to let consumers fall through to the proper libevent headers.
# Note: mindspore_add_pkg is a function so ompi_BASE_DIR doesn't leak out;
# use ompi_DIRPATH (cache var) which it does export.
file(REMOVE_RECURSE ${ompi_DIRPATH}/include/event2)
file(REMOVE
        ${ompi_DIRPATH}/include/event.h
        ${ompi_DIRPATH}/include/evdns.h
        ${ompi_DIRPATH}/include/evhttp.h
        ${ompi_DIRPATH}/include/evrpc.h
        ${ompi_DIRPATH}/include/evutil.h)

include_directories(${ompi_INC})
add_library(mindspore::ompi ALIAS ompi::mpi)