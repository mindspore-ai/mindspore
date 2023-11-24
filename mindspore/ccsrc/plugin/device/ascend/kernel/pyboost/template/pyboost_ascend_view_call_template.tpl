  MS_LOG(DEBUG) << "View Call start";
  runtime::OpExecutor::GetInstance().WaitAll();
  MS_LOG(DEBUG) << "Wait finished";
  auto storage_info_list = ops::${op_name}Calc(primitive_, {${call_args}});
  if (!storage_info_list.empty()) {
    storage_info_list[0]->data_type = ${input}->data_type();
    PrepareOpInputs(device_context_, ${call_tensors});
    PyBoostUtils::CreateOutputTensor(${input}, storage_info_list[0], &outputs_);
    output_abs_ = output(0)->ToAbstract();
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive_->name() <<" or input ERROR";
  }
  return output(0);