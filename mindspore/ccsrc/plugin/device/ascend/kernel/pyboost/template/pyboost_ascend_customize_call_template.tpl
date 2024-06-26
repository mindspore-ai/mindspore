  ProfileMemoryInfo();
  ${customize_func}(get_op(), ${call_args});
  get_op()->CreateOutputSimpleInfoForView();
  return ${return_values};