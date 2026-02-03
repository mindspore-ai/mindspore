  ProfileTrackerTask();
  ${customize_func}(get_op(), ${call_args});
  static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (sync && !device_context_->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name();
  }
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});
  return ${return_values};