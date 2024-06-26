MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});
// ValueTuple to std::vector
${value_tuple_convert}
// Convert ValuePtr to c++ scalar
${const_number_convert}

auto op = get_op();
${create_input_address}
${inplace_process}

PyBoostUtils::PrepareOpOutputs(device_context_, op->stream_id(), outputs_);
ProfileMemoryInfo();

// Async
PyBoostUtils::DispatchRun(
std::make_shared<runtime::PyBoostDeviceTask>(
  [op, ${real_call_args}]() {
    MS_LOG(DEBUG) << "Run device task " << op_name() << " end";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    ${malloc_inputs}
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    ${get_cube_math_type}
    LAUNCH_ACLNN(${aclnn_name}, device_context, op->stream_id(), ${real_call_args}${outputs}${cube_math_type});
    MS_LOG(DEBUG) << "Run device task " << op_name() << " end";
  }
)
);
op->CreateOutputSimpleInfoForView();
return ${return_values};
