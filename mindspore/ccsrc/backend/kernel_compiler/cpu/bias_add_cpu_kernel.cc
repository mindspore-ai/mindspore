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

#include "backend/kernel_compiler/cpu/bias_add_cpu_kernel.h"

#include <functional>

namespace mindspore {
namespace kernel {
constexpr size_t kBiasAddMinDim = 2;
constexpr size_t kBiasAddMaxDim = 5;
constexpr size_t kBiasAddInputNum = 2;

void BiasAddCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  bias_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  bias_param_.ndim_ = input_shape_.size();
  if (bias_param_.ndim_ < kBiasAddMinDim || bias_param_.ndim_ > kBiasAddMaxDim) {
    MS_LOG(EXCEPTION) << "Input tensor's rank must be in closed interval [2,5] for 'BiasAdd' Op,"
                         "but input tensor's rank is "
                      << bias_param_.ndim_;
  }
  if (bias_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "Bias's rank must be 1 for 'BiasAdd' Op, but bias' rank is" << bias_shape_.size();
  }
  if (input_shape_[bias_param_.ndim_ - 1] != bias_shape_[0]) {
    MS_LOG(EXCEPTION) << "Bias shape [" << bias_shape_[0] << "] not match, it must equal C channel's shape:["
                      << input_shape_[bias_param_.ndim_ - 1] << "]";
  }

  for (size_t i = 0; i < bias_param_.ndim_; ++i) {
    bias_param_.in_shape0_[i] = input_shape_[i];
    bias_param_.in_shape1_[i] = 1;
    bias_param_.out_shape_[i] = input_shape_[i];
  }

  bias_param_.in_shape1_[bias_param_.ndim_ - 1] = input_shape_[bias_param_.ndim_ - 1];
}

bool BiasAddCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != kBiasAddInputNum || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "inputs outputs size not supoort";
  }

  auto src_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto bias_addr = reinterpret_cast<float *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  size_t data_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1LL, std::multiplies<int>());

  std::vector<float> buffer_in(data_num, 0);
  std::vector<float> buffer_bias(data_num, 0);
  float *tile_in = &buffer_in.at(0);
  float *tile_bias = &buffer_bias.at(0);

  // BroadcastAdd always returns NNACL_OK, so no need to check return val.
  (void)BroadcastAdd(src_addr, bias_addr, tile_in, tile_bias, output_addr, data_num, &bias_param_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
