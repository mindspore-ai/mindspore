  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});

  // Convert ValueTuple to std::vector
  ${value_tuple_convert}
  // Convert ValuePtr to c++ scalar
  ${const_number_convert}

  // Async
  auto op = get_op();
  DispatchRun(
    std::make_shared<pynative::PyBoostDeviceTask>(
      [op, ${aclnn_call_args}]() {
        auto device_context = op->device_context();
        const auto &outputs = op->outputs();
        // Malloc for input tensors
        ${malloc_inputs}
        // Malloc for output tensors
        PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
        ${op_name}AscendCall(op->primitive(), device_context, ${aclnn_call_args}, outputs);
      }
    )
  );
  return ${return_values};