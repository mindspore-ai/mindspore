  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});
  // ValueTuple to std::vector
  ${value_tuple_convert}
  // Int64Imm to int64_t
  ${const_number_convert}
  // Malloc for input tensors
  ${malloc_inputs}
  ${inplace_process}
  // Malloc for output tensors
  PrepareOpOutputs(device_context_, outputs_);
  // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
  ${get_cube_math_type}
  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  EXEC_NPU_CMD(${aclnn_name}, stream_ptr, ${aclnn_call_args}${outputs}${cube_math_type});
  MS_LOG(DEBUG) << "Launch end";
  return ${return_values};