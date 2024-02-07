  MS_LOG(DEBUG) << "View ${op_name} Call start";
  auto storage_info_list = ops::${op_name}Calc(primitive_, {${call_args}});
  if (!storage_info_list.empty()) {
    // Create device address for input tensors
    PyBoostUtils::PrepareOpInputs(device_context_, ${call_tensors});
    PyBoostUtils::CreateOutputTensor(device_context_, ${input}, storage_info_list[0], &outputs_);

    // Async
    auto op = get_op();
    PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>(
        [op, ${call_tensors}](){
          MS_LOG(DEBUG) << "View device task ${op_name} start";
          auto device_context = op->device_context();
          PyBoostUtils::MallocOpInputs(device_context, ${call_tensors});
          MS_LOG(DEBUG) << "View device task ${op_name} end";
        }
      )
    );

    // Need input abstract to generate grad op.
    if (grad_func_ != nullptr) {
      GenerateAbstract(${call_args});
    }
    output_abs_ = output(0)->ToAbstract();
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive_->name() <<" or input ERROR";
  }
  MS_LOG(DEBUG) << "View ${op_name} Call end";
  return output(0);