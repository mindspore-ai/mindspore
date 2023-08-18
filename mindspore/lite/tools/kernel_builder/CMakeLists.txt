if(MSLITE_ENABLE_ACL AND (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE
                          OR MSLITE_ENABLE_CLOUD_INFERENCE))
  add_subdirectory(ascend)
endif()
