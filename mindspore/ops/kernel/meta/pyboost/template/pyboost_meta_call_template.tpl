MS_LOG(DEBUG) << op_name() << " call start";

InferOutput(${call_args});

CheckInferShapeDynamic(op_name(), outputs_);

// Create device address for output tensors
PyBoostUtils::PrepareOpOutputs(device_context_, 0, outputs_);
MS_LOG(DEBUG) << op_name() << " call end";

return ${return_values};