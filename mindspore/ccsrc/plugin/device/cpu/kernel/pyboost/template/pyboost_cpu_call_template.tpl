MS_LOG(DEBUG) << op_name() << " call start";

InferOutput(${call_args});

${tensor_list_convert}

// Create device address for input tensors
${create_input_address}
${inplace_process}
// Create device address for output tensors
PyBoostUtils::PrepareOpOutputs(device_context_, outputs_);

// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>([this, op, ${call_args_with_tensor}]() {
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();

  // Malloc for input tensors
  ${malloc_inputs}
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  // Get inputs kernel tensors, the not-tensor value will malloc here
  ${get_inputs_kernel_tensors}

  // Get outputs kernel tensors
  const auto &output_address_info =
    PyBoostUtils::GetAddressInfo(device_context, {op->output_abs()}, outputs);

  // KernelMod init
  auto kernel_mod = PyBoostUtils::CreateKernelMod(primitive(), primitive()->name(), op->device_context(),
                                                  input_address_info.first, output_address_info.first);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // KernelMod resize
  if (kernel_mod->Resize(input_address_info.first, output_address_info.first) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << op_name() << "] resize failed.";
  }
  // Get workspace address
  const auto &workspace_device_address = PyBoostUtils::CreateWorkSpaceDeviceAddress(kernel_mod, device_context, op_name());
  const auto &workspace_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(workspace_device_address);
  // Do kernel launch
  if (!kernel_mod->Launch(input_address_info.first, workspace_kernel_tensors, output_address_info.first, nullptr)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << op_name();
  }
  MS_LOG(DEBUG) << "Launch end";
}
)
);
MS_LOG(DEBUG) << op_name() << " call end";
return ${return_values};
