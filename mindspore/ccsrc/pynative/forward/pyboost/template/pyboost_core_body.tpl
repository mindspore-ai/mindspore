PyObject* ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  op_run_info->source_type = source_type;
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, ${has_side_effect}>();
  auto promises = tensor::TransformPromise(py_output);

  DispatchOp(
    std::make_shared<PyboostPromiseTask>(
      [${op_args}, prim, promises](const PyboostOpRunInfoPtr &op_run_info) {

        // stub tensor to tensor.
        ${convert_stub}
        ${implicit_cast}
        kernel::pyboost::OpRunStatus::Get().set_run_info(
            kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                      op_run_info->device_target));
        kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);

        auto outputs = kernel::pyboost::${operator_name}(${cast_args});
        (void)kernel::pyboost::OpRunStatus::Get().GetLastOp();
        tensor::SetPromise(promises, outputs);
      }, [promises]() {
        tensor::SetException(promises);
      }, op_run_info));

    MS_LOG(DEBUG) << "Run ${func_name} end";
    return tensor::TransformOutput(py_output);
}
