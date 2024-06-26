MS_LOG(DEBUG) << op_name() << " call start";
InferOutput(${call_args});

${tensor_list_convert}
MS_EXCEPTION_IF_NULL(primitive());
std::pair<bool, KernelAttr> kernel_attr_pair;
if (output_value_simple_info_ == nullptr) {
  kernel_attr_pair = PyBoostUtils::SelectKernel(input_abs_, output_abs_, device_context_, primitive_->name());
} else {
  AbstractConverter *abs_converter = &kAbstractConverter;
  kernel_attr_pair = PyBoostUtils::SelectKernel(abs_converter, device_context_, primitive_->name(), output_value_simple_info_, ${call_args});
}
if (kernel_attr_pair.first || op_name() == "Cast") {
  // Create device address for input tensors
  auto op = get_op();
  ${create_input_address}
  ${inplace_process}
  // Create device address for output tensors
  PyBoostUtils::PrepareOpOutputs(device_context_, 0, outputs_);
  
  ProfileMemoryInfo();

  // Async
  PyBoostUtils::DispatchRun(
  std::make_shared<runtime::PyBoostDeviceTask>([this, op, ${call_args_with_tensor}]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    ${malloc_inputs}
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // Get inputs kernel tensors, the not-tensor value will malloc here
    ${get_inputs_kernel_tensors}

    // Get outputs kernel tensors
    const auto &output_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    // Launch kernel
    PyBoostUtils::LaunchKernel(primitive(), op->device_context(), input_address_info, output_address_info);
  }));
  MS_LOG(DEBUG) << op_name() << " call end";
  return ${return_values};
}
${cast_input_code}
const auto &op = CREATE_PYBOOST_OP(${op_name_str}, "CPU");
(void)op->Call(${real_call_args_tensor});
std::vector<TypeId> output_types;
for (auto &tensor : outputs()) {
  (void)output_types.emplace_back(tensor->data_type());
}
const auto &real_output = PyBoostUtils::CastTensor(op->outputs(), output_types, "CPU");
set_outputs(real_output);
get_op()->CreateOutputSimpleInfoForView();
return ${return_values};

