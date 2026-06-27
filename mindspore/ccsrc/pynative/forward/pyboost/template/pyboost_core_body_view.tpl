static auto ${func_name}OpExec(${input_args}, bool requires_grad) {
  ${convert_stub}
  return kernel::pyboost::${operator_name}(${call_args});
}

PyObject* ${func_name}OpDispatch(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";

  const auto &pynative_executor = pynative::PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto& forward_executor = pynative_executor->forward_executor();
  const auto &device_target = forward_executor->GetCurrentDeviceTarget(prim);
  bool requires_grad = pynative::GradState::Get().RequiresGrad();

  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(true, device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(requires_grad);
  return tensor::WrapOutput(${func_name}OpExec(${op_args}, requires_grad));
}

PyObject* ${func_name}Op(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  if (IsDispatchEnable()) {
    runtime::Pipeline::Get().WaitFrontend();
    return ${func_name}OpDispatch(prim, source_type, ${op_args});
  }
  MS_LOG(DEBUG) << "Run ${func_name} start";

  const auto &pynative_executor = pynative::PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto& forward_executor = pynative_executor->forward_executor();
  const auto &device_target = forward_executor->GetCurrentDeviceTarget(prim);
  bool requires_grad = pynative::GradState::Get().RequiresGrad();

  // Async path - promise-based dispatch
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, true>();
  auto promises = tensor::TransformPromise(py_output);
  DispatchOp(
    std::make_shared<ViewPyboostPromiseTask>(
      [${op_args}, promises, device_target, requires_grad]() {
        kernel::pyboost::OpRunStatus::Get().set_run_info(
            kernel::pyboost::OpStatus(true, device_target));
        kernel::pyboost::RequireGradGuard require_grad_guard(requires_grad);
        auto outputs = ${func_name}OpExec(${op_args}, requires_grad);
        tensor::SetPromise(promises, outputs);
      },
      [promises]() { tensor::SetException(promises); }));

  MS_LOG(DEBUG) << "Run ${func_name} end";
  return tensor::TransformOutput(py_output);
}
