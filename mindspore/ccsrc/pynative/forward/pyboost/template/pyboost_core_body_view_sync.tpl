PyObject* ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";

  // AsyncStatus
  const auto &pynative_executor = pynative::PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto& forward_executor = pynative_executor->forward_executor();
  const auto &device_target = forward_executor->GetCurrentDeviceTarget(prim);
  bool requires_grad = pynative::GradState::Get().RequiresGrad();

  // stub tensor to tensor.
  ${convert_stub}

  auto py_output_size = CALL_VIEW_INFER_FUNCTION(${op_name}, ${call_args});
  auto py_output = tensor::MakeVector<true>(py_output_size);  
  auto promises = tensor::TransformVectorPromise(py_output);

  DispatchOp(
  std::make_shared<ViewPyboostPromiseTask>(
    [${call_args}, promises, requires_grad, device_target]() {

      kernel::pyboost::OpRunStatus::Get().set_run_info(
        kernel::pyboost::OpStatus(true, device_target));
      kernel::pyboost::RequireGradGuard require_grad_guard(requires_grad);

      auto outputs = kernel::pyboost::${operator_name}(${call_args});

      tensor::SetPromise(promises, outputs);
    },
    [promises]() { tensor::SetException(promises); }));

  MS_LOG(DEBUG) << "Run ${func_name} end";
  return tensor::TransformVectorOutput(py_output);
}
