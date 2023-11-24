  runtime::OpExecutor::GetInstance().WaitAll();
  ${customize_func}(get_op(), ${call_args});
  return ${return_values};