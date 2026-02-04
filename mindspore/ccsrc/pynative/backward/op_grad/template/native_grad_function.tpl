NodePtr NativeFunc::${func_name}(${call_args_with_type}) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kNativeFunc, "${func_name}",
                                     false);
  MS_LOG(DEBUG) << "Begin execute native func" << " ${func_name}";
  if (device_target_ == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Device target is empty!";
  }
#ifndef ENABLE_TEST
  static bool is_kernel_register =
    (kernel::pyboost::PyBoostUtils::IsKernelModRegistered(device_target_, "${func_name}") ||
    kernel::pyboost::PyBoostUtils::IsPyBoostCustomRegistered(device_target_, "${func_name}")) ||
    (ops::IsOpPluginKernel("${func_name}") && device_target_ == device::DeviceType::kCPU);
  if (is_kernel_register) {
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
    auto output_node = std::make_shared<expander::FuncNode>(output_value, output_abs, InputType::kOpOutput, $first_var_name->emitter());

    // Set abstract to tensor cache
    if (op->output_value_simple_info() != nullptr) {
      AutoGradUtil::CacheOutputAbstract(output_value, output_abs);
    }
    MS_LOG(DEBUG) << "End execute native func" << " ${func_name}";
    return output_node;
  }
  auto res = RunOpDeprecated(prim::kPrim${op_name}, {${op_args}});
  MS_LOG(DEBUG) << "End execute native func" << " ${func_name}";
  return res;
#else
  return RunOpInVm(prim::kPrim${op_name}, {${op_args}});
#endif
}
