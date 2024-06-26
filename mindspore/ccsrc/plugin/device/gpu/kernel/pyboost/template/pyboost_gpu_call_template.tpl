MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});

${tensor_list_convert}

auto op = get_op();
// Create device address for input tensors
${create_input_address}
${inplace_process}
// Create device address for output tensors
PyBoostUtils::PrepareOpOutputs(device_context_, op->stream_id(), outputs_);
ProfileMemoryInfo();
// Async
PyBoostUtils::DispatchRun(
std::make_shared<runtime::PyBoostDeviceTask>([this, op, ${call_args_with_tensor}]() {
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();

  // Malloc for input tensors
  ${malloc_inputs}

  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  // Get inputs kernel tensors, the not-tensor value will malloc here
  ${get_inputs_kernel_tensors}

  // Get outputs kernel tensors
  const auto &output_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

  // Launch kernel
  PyBoostUtils::LaunchKernel(primitive(), op->device_context(), input_address_info, output_address_info, op->stream_id());

  // Data sync
  static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (sync && !device_context->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name();
  }
}
)
);
op->CreateOutputSimpleInfoForView();
MS_LOG(DEBUG) << op_name() << " call end";
return ${return_values};
