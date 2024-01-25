MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});
// ValueTuple to std::vector
${value_tuple_convert}
// Convert ValuePtr to c++ scalar
${const_number_convert}

${create_input_address}
${inplace_process}
PyBoostUtils::PrepareOpOutputs(device_context_, outputs_);

// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>(
  [op, ${real_call_args}]() {
      MS_LOG(DEBUG) << "Run device task " << op_name() << " end";
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         op_name(), false);
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      ${malloc_inputs}
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      ${get_cube_math_type}
      auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(op->stream_id());
      LAUNCH_ACLNN(${aclnn_name}, device_context, stream_ptr, ${real_call_args}${outputs}${cube_math_type});
      MS_LOG(DEBUG) << "Run device task " << op_name() << " end";
  }
)
);
return ${return_values};
