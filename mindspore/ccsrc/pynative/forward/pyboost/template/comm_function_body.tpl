${return_type} ${op_name}_dispatch(${input_args_with_type}, CommHandlePtr comm_handle, device::DeviceType target);

${return_type} Dispatch${class_name}(${input_args_with_type}, CommHandlePtr comm_handle, device::DeviceType target) {
  py::gil_scoped_acquire gil;
  auto handle_callback = [&comm_handle, &target](const py::args &args) {
    ${unpack_args}
    return ${op_name}_dispatch(${unpack_call_args}, comm_handle, target);
  };
  auto [packed_args, dispatch_callback, is_guard_callback] = PackArgs(${input_args});
  return HandleMsDispatch(handle_callback, packed_args, dispatch_callback, is_guard_callback, "${class_name}");
}

${return_type} ${op_name}_dispatch(${input_args_with_type}, CommHandlePtr comm_handle, device::DeviceType target) {
  MS_LOG(DEBUG) << "In ${op_name} dispatch function";
  if (ShouldDispatch(${input_args})) {
    return Dispatch${class_name}(${input_args}, comm_handle, target);
  }
  auto op = CREATE_PYBOOST_OP(${class_name}, target);
  op->set_comm_handle(comm_handle);
  
  ${return_type} output;
  OpRunner::CallPreHook("${class_name}", ${input_args});
  {
    PostHookGuard guard;
    output = op->Call(${input_args});
  }
  OpRunner::CallPostHook("${class_name}", output, ${input_args}); 
  return output;
}

${return_type} ${op_name}_inner(${input_args_with_type}, CommHandlePtr comm_handle, device::DeviceType target) {
  MS_LOG(DEBUG) << "In ${op_name} inner function";

  auto output = ${op_name}_dispatch(${input_args}, comm_handle, target);

  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  // clone_tensor is null now.
  if (${op_name}_grad_func) {
    ${op_name}_grad_func(nullptr, ${input_args}, output);
  }
  MS_LOG(DEBUG) << "Out ${op_name} inner function";
  return output;
}

${return_type_with_handle} ${op_name}(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";

  device::DeviceType device_target;
  const auto &group_str = GetValue<std::string>(group);
  if (group_str.compare(0, 4, "mccl") == 0) {
    device_target = device::DeviceType::kCPU;
  } else {
    if (EnableDispatch()) {
      device_target = get_device(${input_args});
    } else {
      device_target = GetDeviceTarget();
    }
  }
  auto comm_handle = std::make_shared<CommHandle>(runtime::OpRunner::GetDeviceContext(device_target));

  comm_handle->CreateEvent();
  auto outputs = ${op_name}_inner(${input_args}, comm_handle, device_target);
  
  MS_LOG(DEBUG) << "Out ${op_name} function";
  return std::make_tuple<${return_type}, CommHandlePtr>(std::move(outputs), std::move(comm_handle));
}
