/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/seq_to_seq_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
bool SeqToSeqCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                  const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_
      << "', dynamic length inputs and outputs address list size must be equal to 1, but got inputs address list size: "
      << inputs.size() << ", outputs address list size: " << outputs.size();
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  auto input_size = inputs[0]->size();
  auto output_size = outputs[0]->size();
  if (input_size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of input[0]: {" << input_size
                      << "} is not equal to the size of output[0]: {" << output_size << "}";
  }
  auto ret = memcpy_s(outputs[0]->device_ptr(), output_size, inputs[0]->device_ptr(), input_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s failed, ret = " << ret;
  }
  return true;
}

std::vector<KernelAttr> SeqToSeqCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  support_list.push_back(KernelAttr().AddSkipCheckAttr(true));
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TupleToList,
                                 []() { return std::make_shared<SeqToSeqCpuKernelMod>(); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ListToTuple,
                                 []() { return std::make_shared<SeqToSeqCpuKernelMod>(); });
}  // namespace kernel
}  // namespace mindspore
