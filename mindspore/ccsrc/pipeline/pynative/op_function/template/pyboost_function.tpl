py::object ${func_name}(const py::args &args) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  auto op_run_info = PyNativeAlgo::PyBoost::Init(args);
  static Parser parser(ops::${op_def_name});
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

      // Create op
      auto op = CREATE_PYBOOST_OP(${op_name}, op_run_info->base_op_run_info.device_target);
      op->set_primitive(op_run_info->op_grad_info->op_prim);

      // Do mixed precision and implicit cast
      PyNativeAlgo::PyBoost::SetCastForInputs({${call_args}}, op_run_info);

      // Run op
      (void)op->Call(${call_args});

      // Update op and op_run_info by op outputs
      PyNativeAlgo::PyBoost::UpdateOpRunInfo(op, {${call_args}}, op_run_info);

      // Do auto grad
      if (op_run_info->requires_grad) {
        op->DoGrad();
      }

      MS_LOG(DEBUG) << "Dispatch ${func_name} end";
    }, op_run_info);
  PyNativeExecutor::GetInstance()->forward_executor()->frontend_queue()->Push(forward_task);
  return node.first;
}
