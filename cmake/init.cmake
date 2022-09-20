
# Attention: cmake will append these flags to compile command automatically.
# So if you want to add global option, change this file rather than flags.cmake

# Linux
# DEBUG:  default: "-g"
# RELEASE:  default: "-O3 -DNDEBUG"
# RELWITHDEBINFO: default: "-O2 -g -DNDEBUG"
# MINSIZEREL: default: "-O2 -g -DNDEBUG"

if(MSVC)
  message("init cxx_flags on windows")
  cmake_host_system_information(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)
  message("CPU_CORE number = ${CPU_CORES}")
  math(EXPR MP_NUM "${CPU_CORES} * 2")
  set(CMAKE_C_FLAGS "/MD /O2 /Ob2 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_C_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_C_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_C_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")

  set(CMAKE_CXX_FLAGS "/std:c++17 /MD /O2 /Ob2 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "/std:c++17 /MDd /Zi /Ob0 /Od /RTC1 /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "/std:c++17 /MD /O2 /Ob2 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/std:c++17 /MD /Zi /O2 /Ob1 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "/std:c++17 /MD /O1 /Ob1 /DNDEBUG /MP${MP_NUM} /EHsc /bigobj")


  # resolve std::min/std::max and opencv::min opencv:max had defined in windows.h
  add_definitions(-DNOMINMAX)
  # resolve ERROR had defined in windows.h
  add_definitions(-DNOGDI)
  add_definitions(-DHAVE_SNPRINTF=1)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4251 /wd4819 /wd4715 /wd4244 /wd4267 /wd4716 /wd4566 /wd4273")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4819 /wd4715 /wd4244 /wd4267 /wd4716 /wd4566 /wd4273")

  if(ENABLE_GPU)
    message("init cxx_flags on windows_gpu")
    set(CMAKE_CUDA_FLAGS "-Xcompiler=\"-MD /MP${MP_NUM} -O2 -Ob2\" -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"-MDd /MP${MP_NUM} -Zi -Ob0 -Od /RTC1\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"-MD /MP${MP_NUM} -O2 -Ob2\" -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-Xcompiler=\"-MD /MP${MP_NUM} -Zi -O2 -Ob1\" -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Xcompiler=\"-MD /MP${MP_NUM} -O1 -Ob1\" -DNDEBUG")
  endif()
endif()
