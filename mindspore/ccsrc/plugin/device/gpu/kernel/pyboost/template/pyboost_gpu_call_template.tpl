MS_LOG(DEBUG) << op_name() << " call start";

InferOutput(${call_args});

const auto &gpu_kernel = kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Create(op_name());
if (gpu_kernel == nullptr) {
  MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Build gpu operator[" << op_name() << "] failed";
}
${value_tuple_convert}
${const_number_convert}
// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>([this, op, gpu_kernel, ${real_call_args}]() {
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();

  // Create device address for inputs
  ${malloc_inputs}
  ${inplace_process}
  // Create device address for outputs
  const auto &outputs_device_address = PyBoostUtils::CreateOutputDeviceAddress(device_context, op->output_abs(), outputs, op->device_sync_promises());
  const auto &outputs_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(outputs_device_address);
  // KernelMod init
  auto ret = gpu_kernel->Init(primitive(), inputs_kernel_tensors, outputs_kernel_tensors);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Init " << op_name() << " failed";
  }
  // KernelMod resize
  if (gpu_kernel->Resize(inputs_kernel_tensors, outputs_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << op_name() << "] resize failed.";
  }
  // Get workspace address
  const auto &workspace_device_address = PyBoostUtils::CreateWorkSpaceDeviceAddress(gpu_kernel, device_context, op_name());
  const auto &workspace_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(workspace_device_address);
  // Do kernel launch
  if (!gpu_kernel->Launch(inputs_kernel_tensors, workspace_kernel_tensors, outputs_kernel_tensors, nullptr)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << op_name();
  }
  MS_LOG(DEBUG) << "Launch end";
}
)
);
MS_LOG(DEBUG) << op_name() << " call end";
return ${return_values};
