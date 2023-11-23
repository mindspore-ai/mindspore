MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});

${tensor_list_convert}

// Create device address for input tensors
${create_input_address}
${inplace_process}
// Create device address for output tensors
PyBoostUtils::PrepareOpOutputs(device_context_, outputs_);

// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>([this, op, ${call_args_with_tensor}]() {
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();

  // Malloc for input tensors
  ${malloc_inputs}

  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  // Get inputs kernel tensors, the not-tensor value will malloc here
  ${get_inputs_kernel_tensors}

  // Get outputs kernel tensors
  const auto &output_address_info =
    PyBoostUtils::GetAddressInfo(device_context, {op->output_abs()}, outputs);

  auto &stream = device::gpu::GPUDeviceManager::GetInstance().GetStream(op->stream_id());
  PyBoostUtils::LaunchKernel(primitive(), op->device_context(),
                             input_address_info, output_address_info, stream);
  static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (sync && !device_context->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name();
  }
}
)
);
MS_LOG(DEBUG) << op_name() << " call end";
return ${return_values};
