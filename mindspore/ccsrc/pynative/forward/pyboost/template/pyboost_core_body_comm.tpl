static auto ${func_name}OpExec(${input_args}, const PyboostOpRunInfoPtr &op_run_info,
    const kernel::pyboost::CommHandlePtr &comm_handle, device::DeviceType target) {
  MS_LOG(DEBUG) << "Run frontend task ${func_name} start";
  ${convert_stub}
  ${implicit_cast}
  auto outputs = kernel::pyboost::${operator_name}_inner(${cast_args}, comm_handle, target);
  MS_LOG(DEBUG) << "Run frontend task ${func_name} end";
  return outputs;
}

PyObject* ${func_name}OpDispatch(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  device::DeviceType target;
  const auto &group_str = GetValue<std::string>(group);
  if (group_str.compare(0, 4, "mccl") == 0) {
    target = device::DeviceType::kCPU;
  } else {
    target = op_run_info->device_target;
  }
  const auto type = GetPredictOutTypeByName(prim->name());
  static auto top_type = GetPredictOutTypeByOutputNum(op_run_info->op_prim, type);

  auto comm_handle_py = std::make_shared<hal::CommHandlePy>(runtime::OpRunner::GetDeviceContext(target));
  auto comm_handle_py_obj = py::cast(comm_handle_py);
  kernel::pyboost::CommHandlePtr comm_handle{nullptr};

  comm_handle_py->comm_handle()->SetOpName(prim->name());
  comm_handle_py->comm_handle()->CreateEvent();
  comm_handle = comm_handle_py->comm_handle();
  op_run_info->source_type = source_type;

  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                op_run_info->device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
  auto outputs = ${func_name}OpExec(${op_args}, op_run_info, comm_handle, target);
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, ${has_side_effect}>();
  auto promises = tensor::TransformPromise(py_output);
  tensor::SetPromise(promises, outputs);
  MS_LOG(DEBUG) << "Run ${func_name} end";
  return tensor::TransformOutput(py_output, comm_handle_py_obj.release().ptr());
}

PyObject* ${func_name}Op(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  if (IsDispatchEnable()) {
    runtime::Pipeline::Get().WaitFrontend();
    return ${func_name}OpDispatch(prim, source_type, ${op_args});
  }
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  device::DeviceType target;
  const auto &group_str = GetValue<std::string>(group);
  if (group_str.compare(0, 4, "mccl") == 0) {
    target = device::DeviceType::kCPU;
  } else {
    target = op_run_info->device_target;
  }
  const auto type = GetPredictOutTypeByName(prim->name());
  static auto top_type = GetPredictOutTypeByOutputNum(op_run_info->op_prim, type);

  auto comm_handle_py = std::make_shared<hal::CommHandlePy>(runtime::OpRunner::GetDeviceContext(target));
  auto comm_handle_py_obj = py::cast(comm_handle_py);
  kernel::pyboost::CommHandlePtr comm_handle{nullptr};

  comm_handle_py->comm_handle()->SetOpName(prim->name());
  comm_handle_py->comm_handle()->CreateEvent();
  comm_handle = comm_handle_py->comm_handle();
  op_run_info->source_type = source_type;

  // Async path - promise-based dispatch
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, ${has_side_effect}>();
  auto promises = tensor::TransformPromise(py_output);
  {
    GilReleaseWithCheck release_gil;
    DispatchOp(
      std::make_shared<PyboostPromiseTask>(
        [${op_args}, promises, comm_handle, target](const PyboostOpRunInfoPtr &op_run_info) {
          kernel::pyboost::OpRunStatus::Get().set_run_info(
              kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                        op_run_info->device_target));
          kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
          auto outputs = ${func_name}OpExec(${op_args}, op_run_info, comm_handle, target);
          tensor::SetPromise(promises, outputs);
        },
        [promises]() {
          tensor::SetException(promises);
        }, op_run_info));
    MS_LOG(DEBUG) << "Run ${func_name} end";
  }
  return tensor::TransformOutput(py_output, comm_handle_py_obj.release().ptr());
}
