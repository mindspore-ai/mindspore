option(MICRO_BUILD_ARM64 "build android arm64" OFF)
option(MICRO_BUILD_ARM32A "build android arm32" OFF)

if(MICRO_BUILD_ARM64 OR MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(MICRO_BUILD_ARM64)
  add_compile_definitions(ENABLE_ARM64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
endif()

if(MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()
