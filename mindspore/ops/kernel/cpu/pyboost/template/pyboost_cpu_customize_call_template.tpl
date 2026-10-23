  if (ops::IsOpPluginKernel(op_name())) {
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
