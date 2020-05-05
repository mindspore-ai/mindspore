/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/bias_add_cpu_kernel.h"

namespace mindspore {
namespace kernel {
void BiasAddCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  bias_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_shape_.size() == 4) {
    data_shape_ = 4;
  } else if (input_shape_.size() == 2) {
    data_shape_ = 2;
  } else {
    MS_LOG(EXCEPTION) << "bias add input data format should be NCHW or NC";
  }
  if (input_shape_.size() != 2 && input_shape_.size() != 4) {
    MS_LOG(EXCEPTION) << "bias add input shape nchw or nc";
  }
  if (bias_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "bias shape invalid";
  }
  if (input_shape_[1] != bias_shape_[0]) {
    MS_LOG(EXCEPTION) << "bias shape not match";
  }
}

bool BiasAddCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                              const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "inputs outputs size not supoort";
  }

  auto src_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto bias_addr = reinterpret_cast<float *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  if (data_shape_ == 4) {
    size_t h_size = input_shape_[3];
    size_t c_size = input_shape_[2] * h_size;
    size_t n_size = input_shape_[1] * c_size;
    size_t hw_size = input_shape_[2] * input_shape_[3];
    size_t n_offset = 0;
    for (size_t n = 0; n < input_shape_[0]; ++n) {
      size_t c_offset = 0;
      for (size_t c = 0; c < input_shape_[1]; ++c) {
        for (size_t hw = 0; hw < hw_size; ++hw) {
          size_t offset = n_offset + c_offset + hw;
          output_addr[offset] = src_addr[offset] + bias_addr[c];
        }
        c_offset += c_size;
      }
      n_offset += n_size;
    }
  } else {
    size_t n_offset = 0;
    for (size_t n = 0; n < input_shape_[0]; ++n) {
      for (size_t c = 0; c < input_shape_[1]; ++c) {
        output_addr[n_offset + c] = src_addr[n_offset + c] + bias_addr[c];
      }
      n_offset += input_shape_[1];
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
