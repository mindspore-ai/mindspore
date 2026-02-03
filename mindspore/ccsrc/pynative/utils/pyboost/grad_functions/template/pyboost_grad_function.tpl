void ${func_name}(OpRunnerInfo* op_runner_info, VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  ${convert_body}
  kernel::pyboost::OpRunStatus::Get().set_run_info(
        kernel::pyboost::OpStatus(true, op_runner_info->device_target));
  auto outputs = kernel::pyboost::${operator_name}(${call_args});
  op_runner_info->output_abs = kernel::pyboost::OpOutputsToAbstract(outputs);
  MS_EXCEPTION_IF_NULL(op_outputs);
  kernel::pyboost::PushOpOutputs(op_outputs, outputs);
}
