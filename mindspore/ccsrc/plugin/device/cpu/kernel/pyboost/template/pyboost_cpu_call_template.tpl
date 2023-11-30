MS_LOG(DEBUG) << op_name() << " call start";

InferOutput(${call_args});
${value_tuple_convert}
${const_number_convert}
// Async
auto op = get_op();
PyBoostUtils::DispatchRun(
std::make_shared<pynative::PyBoostDeviceTask>([this, op, ${real_call_args}]() {
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();

  // Create device address for inputs
  ${malloc_inputs}
  ${inplace_process}
  // Create device address for outputs
  const auto &outputs_device_address = PyBoostUtils::CreateOutputDeviceAddress(device_context, op->output_abs(), outputs, op->device_sync_promises());
  const auto &outputs_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(outputs_device_address);
  // KernelMod init
  auto &cache_helper = kernel::KernelModCache::GetInstance();
  const auto &key = cache_helper.GetKernelModKey(op_name(), "CPU", inputs_kernel_tensors);
  auto kernel_mod = cache_helper.GetKernelMod(key);
  if(kernel_mod == nullptr) {
    kernel_mod = CreateKernelMod(primitive(), op_name(), op->device_context(),
                                 inputs_kernel_tensors, outputs_kernel_tensors);
  }
  const auto &cpu_kernel = std::dynamic_pointer_cast<kernel::NativeCpuKernelMod>(kernel_mod);
  MS_EXCEPTION_IF_NULL(cpu_kernel);
  auto thread_pool = kernel::GetActorMgrInnerThreadPool();
  cpu_kernel->SetThreadPool(thread_pool);
  // KernelMod resize
  if (cpu_kernel->Resize(inputs_kernel_tensors, outputs_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << op_name() << "] resize failed.";
  }
  // Get workspace address
  const auto &workspace_device_address = PyBoostUtils::CreateWorkSpaceDeviceAddress(cpu_kernel, device_context, op_name());
  const auto &workspace_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(workspace_device_address);
  // Do kernel launch
  if (!cpu_kernel->Launch(inputs_kernel_tensors, workspace_kernel_tensors, outputs_kernel_tensors, nullptr)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << op_name();
  }
  MS_LOG(DEBUG) << "Launch end";
}
)
);
MS_LOG(DEBUG) << op_name() << " call end";
return ${return_values};
