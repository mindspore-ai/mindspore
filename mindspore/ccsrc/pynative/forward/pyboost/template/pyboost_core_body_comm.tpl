PyObject* ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
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

    auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}, ${has_side_effect}>();
    auto promises = tensor::TransformPromise(py_output);

    auto comm_handle_py = std::make_shared<hal::CommHandlePy>(runtime::OpRunner::GetDeviceContext(target));
    auto comm_handle_py_obj = py::cast(comm_handle_py);
    kernel::pyboost::CommHandlePtr comm_handle{nullptr};

    comm_handle_py->comm_handle()->CreateEvent();
    comm_handle = comm_handle_py->comm_handle();
    {
        GilReleaseWithCheck release_gil;
        op_run_info->source_type = source_type;
        DispatchOp(
            std::make_shared<PyboostPromiseTask>(
            [${op_args}, comm_handle, target, promises](const PyboostOpRunInfoPtr &op_run_info) {
                MS_LOG(DEBUG) << "Run frontend task ${func_name} start";

               // stub tensor to tensor.
               ${convert_stub}
               ${implicit_cast}

                kernel::pyboost::OpRunStatus::Get().set_run_info(
                            kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                                      op_run_info->device_target));
                kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
                // Run op
                auto outputs = kernel::pyboost::${operator_name}_inner(${cast_args}, comm_handle, target);
                auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();
                tensor::SetPromise(promises, outputs);
                MS_LOG(DEBUG) << "Run frontend task ${func_name} end";
            },
            [promises]() {
              tensor::SetException(promises);
            }, op_run_info)
        );
        MS_LOG(DEBUG) << "Run ${func_name} end";
    }
    return tensor::TransformOutput(py_output, comm_handle_py_obj.release().ptr());
}
