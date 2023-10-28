py::object ${func_name}(const py::args &args) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  auto op_run_info = PyNativeAlgo::PyBoost::Init(args);
  static Parser parser(&ops::${op_def_name});
  py::list input_args = args[kIndex1];
  parser.Parse(input_args);
  ${parser_body}

  auto top_type = PredictOutType(op_run_info);
  auto node = stub::MakeTopNode(top_type);
  GilReleaseWithCheck release_gil;
  op_run_info->stub_output = node.second;
  auto forward_task = std::make_shared<FrontendTask>(
    [${op_args}](const FrontendOpRunInfoPtr &op_run_info) {
      // stub tensor to tensor.
      ${convert_stub}

      auto op = CREATE_PYBOOST_OP(${op_name}, op_run_info->base_op_run_info.device_target);
      op->set_primitive(op_run_info->op_grad_info->op_prim);
      (void)op->Call(${call_args});

      op_run_info->base_op_run_info.abstract = op->output_abs();
      PyNativeAlgo::PyBoost::MakeOutputValue(op_run_info, op->outputs());
      PyNativeAlgo::PyBoost::UpdateStubOutput(op_run_info, op->output_abs());

      if (op_run_info->requires_grad) {
        op->set_grad_func([op_run_info](const std::vector<ValuePtr> &inputs, const std::vector<TensorPtr> &output,
                                        const std::vector<abstract::AbstractBasePtr> &input_abs,
                                        const AbstractBasePtr &output_abs) {
          PyNativeAlgo::PyBoost::DoGrad(op_run_info, inputs, output, input_abs, output_abs);
        });

        op->DoGrad({${do_grad_args}});
      }

      MS_LOG(DEBUG) << "Dispatch ${func_name} end";
    },
    op_run_info);
  PyNativeExecutor::GetInstance()->forward_executor()->frontend_queue()->Push(forward_task);
  return node.first;
}