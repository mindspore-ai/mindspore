void ${func_name}(OpRunnerInfo* op_runner_info, VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  // Create op
  auto op = CREATE_PYBOOST_OP(${op_name}, op_runner_info->device_target);
  op->set_primitive(op_runner_info->prim);

  // Run op
  ${convert_body}
  (void)op->Call(${call_args}, op_runner_info);
  MS_EXCEPTION_IF_NULL(op_outputs);
  (void)std::transform(op->outputs().begin(), op->outputs().end(), std::back_inserter(*op_outputs),
                       [] (const auto &item) {return item;});
}
