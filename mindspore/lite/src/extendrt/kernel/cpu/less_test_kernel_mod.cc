/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>

#include "extendrt/kernel/cpu/less_test_kernel_mod.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore::kernel {
const size_t test_input_size = 2;
const int test_input_shape = 7;

bool LessTestKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_LOG(INFO) << "LessTestKernelMod::Launch";
  // test shape 7 value
  MS_ASSERT(inputs.size() == test_input_size);
  auto x = static_cast<int *>(inputs[0]->addr);
  auto y = static_cast<int *>(inputs[1]->addr);
  auto z = static_cast<bool *>(outputs[0]->addr);

  for (int i = 0; i < test_input_shape; i++) {
    if (x[i] < y[i]) {
      z[i] = true;
    } else {
      z[i] = false;
    }
  }

  for (int i = 0; i < test_input_shape; i++) {
    MS_LOG(INFO) << "LessTestKernelMod::Launch z " << z[i];
  }

  return true;
}

bool LessTestKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_LOG(INFO) << "LessTestKernelMod::Init";
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(CpuKernelMod, Less,
                                 []() { return std::make_shared<LessTestKernelMod>(prim::kPrimLess->name()); });
}  // namespace mindspore::kernel
