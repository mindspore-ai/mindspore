# MS Utils
#

function(find_python_package out_inc out_lib)
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python3")
  else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
  endif()

  execute_process(
          COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE inc)
  string(STRIP "${inc}" inc)
  set(${out_inc} ${inc} PARENT_SCOPE)

  execute_process(
          COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig as sysconfig; import os; \
                  print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE lib)
  string(STRIP "${lib}" lib)
  set(${out_lib} ${lib} PARENT_SCOPE)
endfunction()
