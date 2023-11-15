  runtime::OpExecutor::GetInstance().WaitAll();
  MS_LOG(DEBUG) << "Call start";
  InferOutput(${call_args});
  // ValueTuple to std::vector
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
          ${inplace_process}
          // Malloc for output tensors
          PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
          // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
          ${get_cube_math_type}
          auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
          LAUNCH_ACLNN(${aclnn_name}, device_context, stream_ptr, ${aclnn_call_args}${outputs}${cube_math_type});
          MS_LOG(DEBUG) << "Launch end";
      }
    )
  );
  return ${return_values};