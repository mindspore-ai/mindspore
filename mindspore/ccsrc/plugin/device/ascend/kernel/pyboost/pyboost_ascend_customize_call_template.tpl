  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});

  // Don't need to allocate memory for Scalar.
  DeviceMalloc(${call_tensors});
  return ${op_name}AscendCall(primitive_, ${call_args});