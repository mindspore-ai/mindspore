NodePtr NativeFunc::${func_name}(${call_args_with_type}) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kNativeFunc, "${func_name}",
                                     false);
  MS_LOG(DEBUG) << "Begin execute native func" << " ${func_name}";
  if (device_target_ == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Device target is empty!";
  }
#ifndef ENABLE_TEST
  // Run op
  ${convert_body}
  kernel::pyboost::OpRunStatus::Get().set_run_info(
    kernel::pyboost::OpStatus(true, device_target_));
  auto outputs = kernel::pyboost::${operator_name}(${call_args});
  abstract::AbstractBasePtr output_abs = kNativeAbstractConverter.ConvertAbstract(outputs);
  ${output_expr}
  auto output_node = std::make_shared<expander::FuncNode>(output_value, output_abs, InputType::kOpOutput, $first_var_name->emitter());

  // Set abstract to tensor cache
  AutoGradUtil::CacheOutputAbstract(output_value, output_abs);
  MS_LOG(DEBUG) << "End execute native func" << " ${func_name}";
  return output_node;
#else
  return RunOpInVm(prim::kPrim${op_name}, {${op_args}});
#endif
}
