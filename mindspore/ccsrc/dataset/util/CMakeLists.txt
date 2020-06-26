file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(utils OBJECT
    arena.cc
    buddy.cc
    cache_pool.cc
    circular_pool.cc
    memory_pool.cc
    cond_var.cc
    intrp_service.cc
    task.cc
    task_manager.cc
    service.cc
    services.cc
    lock.cc
    semaphore.cc
    status.cc
    storage_container.cc
    storage_manager.cc
    slice.cc
    path.cc
    wait_post.cc
    sig_handler.cc)
