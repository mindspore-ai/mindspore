${return_type} ${op_name}_with_grad(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";

  device::DeviceType device_target;
  if (EnableDispatch()) {
    device_target = get_device(${input_args});
  } else {
    device_target = GetDeviceTarget();
  }

  OpRunStatus::Get().HeterBarrier(device_target);
  ${create_op}
  ${clone_func}
  auto output = op->Call(${input_args});

  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  if (${op_name}_grad_func){
    ${op_name}_grad_func(op, ${input_args});
  }
  MS_LOG(DEBUG) << "Out ${op_name} function";
  OpRunStatus::Get().SetLastOp(op);
  return output;
}
