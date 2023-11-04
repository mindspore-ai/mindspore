  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});

  // Don't need to allocate memory for Scalar.
  DeviceMalloc(${call_tensors});
  return ${op_name}AscendCall(primitive_, device_context_, ${call_args}, outputs_);