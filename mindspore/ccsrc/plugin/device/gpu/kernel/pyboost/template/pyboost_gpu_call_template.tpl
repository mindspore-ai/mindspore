  MS_LOG(DEBUG) << op_name() << " call start";

  InferOutput(${call_args});

  const auto &gpu_kernel = kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Create(op_name());
  if (gpu_kernel == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Build gpu operator[" << op_name() << "] failed";
  }

  // ValueTuple to std::vector
  ${value_tuple_convert}
  // Convert ValuePtr to c++ scalar
  ${const_number_convert}
  // Async
  auto op = get_op();
  DispatchRun(
      std::make_shared<pynative::PyBoostDeviceTask>(
        [this, op, gpu_kernel, ${aclnn_call_args}]() {
          auto device_context = op->device_context();
          const auto &outputs = op->outputs();
  
          // Create device address for input tensors
          ${malloc_inputs}
          ${inplace_process}
          // Create device address for output tensors
          const auto &output_device_address = PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
          std::vector<kernel::KernelTensor *> outputs_kernel_tensors;
          std::transform(output_device_address.begin(),output_device_address.end(),
            std::back_inserter(outputs_kernel_tensors), [] (const auto &item) {return item->kernel_tensor().get();});
  
          auto ret = gpu_kernel->Init(primitive(), input_kernel_tensors, outputs_kernel_tensors);
          if (!ret) {
           MS_LOG(EXCEPTION) << "Init " << op_name() << " failed";
          }
          if (gpu_kernel->Resize(input_kernel_tensors, outputs_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
              MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << op_name() << "] resize failed.";
          }

          // Get workspace kernel tensors
          auto workspace_kernel_tensors = GetWorkspaceKernelTensors(gpu_kernel, device_context, op_name());
  
          // Do kernel launch
          if (!gpu_kernel->Launch(input_kernel_tensors, workspace_kernel_tensors, outputs_kernel_tensors, nullptr)) {
            MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << op_name();
          }
          MS_LOG(DEBUG) << "Launch end";
        }
      )
    );
  MS_LOG(DEBUG) << op_name() << " call end";
  return ${return_values};
