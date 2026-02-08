NodePtr NativeFunc::${func_name}(${call_args_with_type}) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kNativeFunc, "${func_name}",
                                     false);
  MS_LOG(DEBUG) << "Begin execute native comm func" << " ${func_name}";
  if (device_target_ == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Device target is empty!";
  }
#ifndef ENABLE_TEST
  // Run op
  ${convert_body}
  kernel::pyboost::OpRunStatus::Get().set_run_info(
    kernel::pyboost::OpStatus(true, device_target_));
  auto outputs = kernel::pyboost::${operator_name}(${call_args});
  auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();
  abstract::AbstractBasePtr output_abs;
  if (op->output_value_simple_info() != nullptr) {
      // Get output abstract
      output_abs = TransformValueSimpleInfoToAbstract(*op->output_value_simple_info());
  } else {
    MS_EXCEPTION_IF_NULL(op->output_abs());
    output_abs = op->output_abs();
  }
  ${output_expr}
  auto output_node = std::make_shared<CommFuncNode>(output_value, output_abs, InputType::kOpOutput, $first_var_name->emitter(), std::get<1>(outputs));
  // Set abstract to tensor cache
  MS_LOG(DEBUG) << "End execute native comm func" << " ${func_name}";
  return output_node;
#else
  return RunOpInVm(prim::kPrim${op_name}, {${op_args}});
#endif
}

