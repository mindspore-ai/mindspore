  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});

  // Don't need to allocate memory for Scalar.
  DeviceMalloc(${call_tensors});
  ${value_tuple_convert}
  ${const_number_convert}
  ${get_cube_math_type}
  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  EXEC_NPU_CMD(${aclnn_name}, stream_ptr, ${aclnn_call_args}, ${outputs}${cube_math_type});
  MS_LOG(DEBUG) << "Launch end";
  // return TensorPtr or std::tuple(TensorPtr) or self define function.
  return ${return_values};