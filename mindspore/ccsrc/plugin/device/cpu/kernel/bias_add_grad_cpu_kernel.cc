/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/bias_add_grad_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/reduce_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBiasAddGradInputsNum = 1;
constexpr size_t kBiasAddGradOutputsNum = 1;
}  // namespace

void BiasAddGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (IsDynamic(shape)) {
    return;
  }

  input_shape_ = Convert2SizeT(shape);
  if (input_shape_.size() < 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor's dimension must be at least 2, but got "
                      << input_shape_.size();
  }
}

bool BiasAddGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBiasAddGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBiasAddGradOutputsNum, kernel_name_);
  const auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  if (input_shape_.size() > 2) {
    size_t hw_size = 1;
    for (size_t i = 2; i < input_shape_.size(); ++i) {
      hw_size *= input_shape_[i];
    }

    size_t c_size = input_shape_[1];
    for (size_t c = 0; c < c_size; ++c) {
      output_addr[c] = 0;
      for (size_t n = 0; n < input_shape_[0]; ++n) {
        size_t offset = n * c_size * hw_size + c * hw_size;
        for (size_t hw = 0; hw < hw_size; ++hw) {
          output_addr[c] += input_addr[offset + hw];
        }
      }
    }
  } else if (input_shape_.size() == 2) {
    auto task = [this, input_addr, output_addr](size_t start, size_t end) {
      int ret =
        ReduceSumDim2Axis0(end - start, input_shape_[1], input_shape_[0], input_addr + start, output_addr + start);
      if (ret != NNACL_OK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', ReduceSumDim2Axis0 failed. Error no: " << ret;
      }
    };
    ParallelLaunchAutoSearch(task, input_shape_[1], this, &parallel_search_info_);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BiasAddGrad, BiasAddGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
