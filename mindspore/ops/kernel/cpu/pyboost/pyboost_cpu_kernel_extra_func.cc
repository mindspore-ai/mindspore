/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mindspore/ops/kernel/cpu/pyboost/pyboost_cpu_kernel_extra_func.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "kernel/cpu/cpu_kernel.h"
#include "utils/ms_context.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ops/kernel/cpu/pyboost/pyboost_cpu_custom_kernel_register.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/contiguous.h"
#include "ops/op_def.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MakeCpuFallbackContiguous(const tensor::TensorPtr &input) {
  const auto &op = CREATE_PYBOOST_OP(Contiguous, device::DeviceType::kCPU);
  return op->Call(input);
}

// View operators retain their existing input-layout policy.
bool PrepareCpuFallbackEnabled(const std::string &name) {
  if (name == "Contiguous" || name == "Copy") {
    return false;
  }
  const auto op_def = ops::GetOpDef(name);
  return op_def == nullptr || !op_def->is_view_;
}

tensor::TensorPtr PrepareCpuFallbackInput(const tensor::TensorPtr &input, bool *changed) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->is_contiguous()) {
    return input;
  }
  auto output = MakeCpuFallbackContiguous(input);
  MS_EXCEPTION_IF_NULL(output);
  if (!output->is_contiguous()) {
    MS_LOG(EXCEPTION) << "CPU Contiguous returned a non-contiguous tensor.";
  }
  *changed = true;
  return output;
}

ValuePtr PrepareCpuFallbackInput(const ValuePtr &input, bool *changed) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::Tensor>()) {
    return PrepareCpuFallbackInput(input->cast<tensor::TensorPtr>(), changed);
  }
  if (input->isa<ValueTuple>()) {
    return PrepareCpuFallbackInput(input->cast<ValueTuplePtr>(), changed);
  }
  return input;
}

ValueTuplePtr PrepareCpuFallbackInput(const ValueTuplePtr &input, bool *changed) {
  MS_EXCEPTION_IF_NULL(input);
  auto values = input->value();
  bool tuple_changed = false;
  for (auto &value : values) {
    value = PrepareCpuFallbackInput(value, &tuple_changed);
  }
  if (!tuple_changed) {
    return input;
  }
  *changed = true;
  return std::make_shared<ValueTuple>(std::move(values));
}

void PyboostCPUKernelExtraFunc::SetThreadPool(const kernel::KernelModPtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);

  const auto &cpu_kernel = std::dynamic_pointer_cast<kernel::NativeCpuKernelMod>(kernel);
  MS_EXCEPTION_IF_NULL(cpu_kernel);
  auto thread_pool = kernel::GetActorMgrInnerThreadPool();
  cpu_kernel->SetThreadPool(thread_pool);
}

bool PyboostCPUKernelExtraFunc::IsKernelModRegistered(const std::string &op_name) {
  return kernel::Factory<kernel::NativeCpuKernelMod>::Instance().IsRegistered(op_name);
}

bool PyboostCPUKernelExtraFunc::IsPyBoostCustomRegistered(const std::string &op_name) {
  return PyBoostCpuCustomKernel::GetInstance().IsPyBoostCustomRegistered(op_name);
}

bool PyboostCPUKernelExtraFunc::IsEnableProfiler() {
  const auto &profiler_inst = mindspore::profiler::Profiler::GetInstance(kCPUDevice);
  MS_EXCEPTION_IF_NULL(profiler_inst);
  return profiler_inst->GetEnableFlag() && profiler_inst->GetOpTimeFlag();
}

void PyboostCPUKernelExtraFunc::LaunchKernelWithProfiler(const std::string &op_name,
                                                         const device::DeviceContext *device_context,
                                                         const std::vector<BaseShapePtr> &base_shape,
                                                         const std::function<void()> &func) {
  auto profiler_inst = mindspore::profiler::Profiler::GetInstance(kCPUDevice);
  MS_EXCEPTION_IF_NULL(profiler_inst);

  uint32_t pid = IntToUint(getpid());
  // cpu support multi-thread with mindrt for profiling.
  profiler_inst->OpDataProducerBeginParallel(op_name, pid);
  // launch kernel.
  func();
  profiler_inst->OpDataProducerEndParallel(op_name);
  profiler_inst->RecordFrameWorkInfo(op_name, base_shape);
}

REG_PYBOOST_KERNEL_EXTRA_FUN(CPU, PyboostCPUKernelExtraFunc);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
