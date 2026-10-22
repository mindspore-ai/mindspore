  // The plugin registry and library handle are initialized once per process.
  // Each generated Call method belongs to one fixed op name, so cache the route
  // decision instead of hashing and looking up the same name on every call.
  static const bool use_op_plugin = ops::IsOpPluginKernel(op_name());
  if (use_op_plugin) {
    ProfileTrackerTask();
    outputs_ = PyboostLaunchOpPluginKernel<${inplace_indices}>(get_op(), ${call_args});
    return ${return_values};
  }
  static const bool prepare_cpu_inputs = PrepareCpuFallbackEnabled(op_name());
  if (prepare_cpu_inputs) {
    bool changed = false;
    auto inputs = PrepareCpuFallbackInputs(std::make_tuple(${call_args}), &changed);
    if (changed) {
      return std::apply([this](const auto &...args) { return this->Call(args...); }, inputs);
    }
  }
  ProfileTrackerTask();
  ${customize_func}(get_op(), ${call_args});
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});
  return ${return_values};
