/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/allreduce_cpu_kernel.h"

#include <set>
#include <functional>
#include <memory>

#if defined(__linux__) && defined(WITH_BACKEND)
#include "plugin/device/cpu/hal/hardware/ms_collective_comm_lib.h"
#endif

namespace mindspore {
namespace kernel {
#if defined(__linux__) && defined(WITH_BACKEND)
using device::CollectiveOpReduceType::Reduce_Sum;
using device::cpu::kMCCLGlobalGroupName;
using device::cpu::MsCollectiveCommLib;
#endif

namespace {
constexpr char kSupportedReduceOp[] = "sum";
}  // namespace

bool AllReduceCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
#if defined(__linux__) && defined(WITH_BACKEND)
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  auto group = GetValue<std::string>(prim->GetAttr(GROUP));
  if (group != kMCCLGlobalGroupName) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only support " << kMCCLGlobalGroupName << " on CPU, but got " << group;
  }
  auto reduce_op = GetValue<std::string>(prim->GetAttr(OP));
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
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

bool AllReduceCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << kernel_name_ << " has at least one input and one output, but got 0.";
  }
  std::size_t data_size = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    data_size += inputs[i]->size;
  }
  bool ret = MsCollectiveCommLib::GetInstance().AllReduce(inputs[0]->addr, outputs[0]->addr, data_size,
                                                          kNumberTypeFloat32, Reduce_Sum, kMCCLGlobalGroupName);
  if (!ret) {
    MS_LOG(ERROR) << "AllReduceCPUKernelMod launch failed.";
  }
  return ret;
#else
  MS_LOG(EXCEPTION) << "The CPU kernel allreduce is only supported on linux platform.";
#endif
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AllReduce, AllReduceCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
