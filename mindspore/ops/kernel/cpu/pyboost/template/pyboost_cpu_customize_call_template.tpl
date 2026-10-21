  ProfileTrackerTask();
  // The plugin registry and library handle are initialized once per process.
  // Each generated Call method belongs to one fixed op name, so cache the route
  // decision instead of hashing and looking up the same name on every call.
  static const bool use_op_plugin = ops::IsOpPluginKernel(op_name());
  if (use_op_plugin) {
    outputs_ = PyboostLaunchOpPluginKernel<${inplace_indices}>(get_op(), ${call_args});
    return ${return_values};
  }
  ${customize_func}(get_op(), ${call_args});
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});
  return ${return_values};