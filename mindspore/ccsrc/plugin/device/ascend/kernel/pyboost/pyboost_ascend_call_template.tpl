  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});
  // Don't need to allocate memory for Scalar.
  DeviceMalloc(${call_tensors});
  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  ${launch_mode}(aclnn${op_name}, stream_ptr, ${call_args}, ${outputs});
  MS_LOG(DEBUG) << "Launch end";
  // return TensorPtr or std::tuple(TensorPtr) or self define function.
  return ${return_values};