NodePtr NativeFunc::${func_name}(${call_args_with_type}) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kNativeFunc, "${func_name}",
                                     false);
  if (device_target_.empty()) {
    MS_LOG(EXCEPTION) << "Device target is empty!";
  }
#ifndef ENABLE_TEST
  static bool is_kernel_register =
    (kernel::pyboost::PyBoostUtils::IsKernelModRegistered(device_target_, "${func_name}") ||
    kernel::pyboost::PyBoostUtils::IsPyBoostCustomRegistered(device_target_, "${func_name}"));
  if (is_kernel_register) {
    // Create op
    auto op = CREATE_PYBOOST_OP(${op_name}, device_target_);
    op->set_primitive(prim::kPrim${op_name});

    // Run op
    ${convert_body}
    (void)op->Call(${call_args});
    op->CreateOutputSimpleInfoForView();
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
      PyNativeAlgo::AutoGrad::CacheOutputAbstract(output_value, output_abs);
    }
    return output_node;
  }
  return RunOpDeprecated(prim::kPrim${op_name}, {${op_args}});
#else
  return RunOpInVm(prim::kPrim${op_name}, {${op_args}});
#endif
}
