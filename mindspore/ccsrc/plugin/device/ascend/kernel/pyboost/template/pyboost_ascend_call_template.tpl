runtime::OpExecutor::GetInstance().WaitAll();
MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});
${value_tuple_convert}
${const_number_convert}
// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>(
  [op, ${real_call_args}]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    ${malloc_inputs}
    ${inplace_process}
    // Malloc for output tensors
    PyBoostUtils::PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
    ${get_cube_math_type}
    auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
    LAUNCH_ACLNN(${aclnn_name}, device_context, stream_ptr, ${real_call_args}${outputs}${cube_math_type});
    MS_LOG(DEBUG) << "Launch end";
  }
)
);
return ${return_values};