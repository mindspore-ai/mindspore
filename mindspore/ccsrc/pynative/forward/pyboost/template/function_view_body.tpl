namespace {
inline static ${return_type} ${op_name}_inner(${call_args_with_type}) {
  MS_LOG(DEBUG) << "View ${op_name} Call start";

  device::DeviceType device_target;
  if (EnableDispatch()) {
    device_target = get_device(${call_args});
  } else {
    device_target = GetDeviceTarget();
  }
  OpRunStatus::Get().HeterBarrier(device_target);

  const auto &device_context = runtime::OpRunner::GetDeviceContext(device_target);
  auto cur_stream_id = CurrentStream::id();

  tensor::TensorPtrList outputs;
  auto view_info = ops::${storage_calc}BasicTypeCalc(${call_args});

  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(device_context, cur_stream_id, ${call_tensors});
  PyBoostUtils::CreateOutputTensor(device_context, ${input}, view_info, &outputs);

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>(
      [device_context, ${call_tensors}](){
        MS_LOG(DEBUG) << "View device task ${op_name} start";
        PyBoostUtils::MallocOpInputs(device_context, ${call_tensors});
        MS_LOG(DEBUG) << "View device task ${op_name} end";
      }
    )
  );

  MS_LOG(DEBUG) << "View ${op_name} Call end";
  return ${return_values};
}
} //  namespace

${return_type} ${op_name}(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrim${class_name});

  auto output = ${op_name}_inner(${input_args});

  ProfileTracker::ProfileTrackerInput(prim::kPrim${class_name}, skip_tracker, ${input_args});
  ProfileTracker::ProfileTrackerOutput(prim::kPrim${class_name}, skip_tracker, output);

  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  ${op_name}_grad_func(output, ${input_args});

  MS_LOG(DEBUG) << "Out ${op_name} function";
  return output;
}
