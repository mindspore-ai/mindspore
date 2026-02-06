${return_type} ${op_name}_inner(${input_args_with_type}, CommHandlePtr comm_handle, device::DeviceType target) {
  MS_LOG(DEBUG) << "In ${op_name} inner function";

  auto op = CREATE_PYBOOST_OP(${class_name}, target);
  op->set_comm_handle(comm_handle);

  auto output = op->Call(${input_args});
  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  ${op_name}_grad_func(op, ${input_args});
  MS_LOG(DEBUG) << "Out ${op_name} inner function";
  OpRunStatus::Get().SetLastOp(op);
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
