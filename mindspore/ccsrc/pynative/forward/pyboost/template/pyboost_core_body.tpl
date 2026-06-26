static auto ${func_name}OpExec(${input_args}, const PyboostOpRunInfoPtr &op_run_info) {
  ${convert_stub}
  ${implicit_cast}
  return kernel::pyboost::${operator_name}(${cast_args});
}

PyObject* ${func_name}OpDispatch(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  op_run_info->source_type = source_type;

  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                op_run_info->device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
  return tensor::WrapOutput(${func_name}OpExec(${op_args}, op_run_info));
}

PyObject* ${func_name}Op(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  if (IsDispatchEnable()) {
    runtime::Pipeline::Get().WaitFrontend();
    return ${func_name}OpDispatch(prim, source_type, ${op_args});
  }
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  op_run_info->source_type = source_type;

  // Async path - promise-based dispatch
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, ${has_side_effect}>();
  auto promises = tensor::TransformPromise(py_output);
  DispatchOp(
    std::make_shared<PyboostPromiseTask>(
      [${op_args}, promises](const PyboostOpRunInfoPtr &op_run_info) {
        kernel::pyboost::OpRunStatus::Get().set_run_info(
            kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                      op_run_info->device_target));
        kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
        auto outputs = ${func_name}OpExec(${op_args}, op_run_info);
        tensor::SetPromise(promises, outputs);
      }, [promises]() {
        tensor::SetException(promises);
      }, op_run_info));

  MS_LOG(DEBUG) << "Run ${func_name} end";
  return tensor::TransformOutput(py_output);
}
