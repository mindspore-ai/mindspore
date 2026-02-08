/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/allreduce_cpu_kernel.h"

#include <set>
#include <functional>
#include <memory>
#include "utils/misc.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "plugin/cpu/res_manager/collective/ms_collective_comm_lib.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#endif

namespace mindspore {
namespace kernel {
namespace allreduce_cpu {
#if defined(__linux__) && defined(WITH_BACKEND)
using device::CollectiveOpReduceType::Reduce_Sum;
using device::cpu::kMCCLGlobalGroupName;
#endif

namespace {
constexpr char kSupportedReduceOp[] = "sum";
}  // namespace

bool AllReduceCPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
#if defined(__linux__) && defined(WITH_BACKEND)
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  auto group = GetValue<std::string>(primitive_->GetAttr(GROUP));
  group_ = group;
  input_dtype_ = inputs[0]->dtype_id();
  auto reduce_op = GetValue<std::string>(primitive_->GetAttr(OP));
  if (reduce_op != kSupportedReduceOp) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only support reduce sum on CPU, but got " << reduce_op;
  }
#else
  MS_LOG(EXCEPTION) << "The CPU kernel allreduce is only supported on linux platform.";
#endif
  return true;
}

std::vector<KernelAttr> AllReduceCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

bool AllReduceCPUKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << kernel_name_ << " has at least one input and one output, but got 0.";
  }
  std::size_t data_size = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    data_size += inputs[i]->size();
  }
  auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
  data_size = data_size / GetDataTypeSize(input_dtype_);
  if (data_size == 0) {
    MS_LOG(DEBUG) << "AllReduceCPUKernelMod: data_size is 0, skip AllReduce operation.";
    return true;
  }
  bool ret =
    comm_lib->AllReduce(inputs[0]->device_ptr(), outputs[0]->device_ptr(), data_size, input_dtype_, Reduce_Sum, group_);
  if (!ret) {
    MS_LOG(ERROR) << "AllReduceCPUKernelMod launch failed.";
  }
  return ret;
#else
  MS_LOG(EXCEPTION) << "The CPU kernel allreduce is only supported on linux platform.";
#endif
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AllReduce, AllReduceCPUKernelMod);
}  // namespace allreduce_cpu
}  // namespace kernel
}  // namespace mindspore
