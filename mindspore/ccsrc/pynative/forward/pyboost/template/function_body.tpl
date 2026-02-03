${return_type} ${op_name}_impl(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} impl function";
  if (ShouldDispatch(${input_args})) {
    py::gil_scoped_acquire gil;
    auto handle_callback = [](const py::args &args) {
      ${unpack_args}
      return ${op_name}_impl(${unpack_call_args});
    };
    auto [packed_args, dispatch_callback, is_guard_callback] = PackArgs(${input_args});
    return HandleMsDispatch(handle_callback, packed_args, dispatch_callback, is_guard_callback, "${class_name}");
  }
  device::DeviceType device_target = EnableDispatch() ? get_device(${input_args}) : GetDeviceTarget();
  OpRunStatus::Get().HeterBarrier(device_target);
  ${create_op}
  return op->Call(${input_args});
}

${return_type} ${op_name}(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";
  ${clone_func}
  auto output = ${op_name}_impl(${input_args});
  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  if (${op_name}_grad_func){
    ${op_name}_grad_func(clone_tensor, ${input_args}, output);
  }
  MS_LOG(DEBUG) << "Out ${op_name} function";
  return output;
}

