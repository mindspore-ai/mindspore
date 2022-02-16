/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class LayerNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  LayerNormGpuKernelMod() : input_row_(1), input_col_(1), param_dim_(1), is_null_input_(false) {}
  ~LayerNormGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto gamma = GetDeviceAddress<T>(inputs, 1);
    auto beta = GetDeviceAddress<T>(inputs, 2);
    auto y = GetDeviceAddress<T>(outputs, 0);
    auto mean = GetDeviceAddress<T>(outputs, 1);
    auto variance = GetDeviceAddress<T>(outputs, 2);

    const T epsilon = 10e-12;
    LayerNorm(input_row_, input_col_, param_dim_, epsilon, x, gamma, beta, y, mean, variance,
              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    int begin_norm_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_norm_axis"));
    int begin_params_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_params_axis"));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input_x");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (begin_norm_axis < 0) {
      begin_norm_axis += input_shape.size();
    }

    if (begin_params_axis < 0) {
      begin_params_axis += input_shape.size();
    }

    if (IntToSize(begin_norm_axis) > input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of 'begin_norm_axis' should be less than or equal "
                        << "to the dimension of input_x, but got begin_norm_axis: " << IntToSize(begin_norm_axis)
                        << ", the dimension of input_x: " << input_shape.size();
    }
    for (size_t i = 0; i < IntToSize(begin_norm_axis); i++) {
      input_row_ *= input_shape[i];
    }

    for (size_t i = begin_norm_axis; i < input_shape.size(); i++) {
      input_col_ *= input_shape[i];
    }

    for (size_t i = begin_params_axis; i < input_shape.size(); i++) {
      param_dim_ *= input_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_row_ * input_col_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));

    output_size_list_.push_back(input_row_ * input_col_ * sizeof(T));
    output_size_list_.push_back(input_row_ * sizeof(T));
    output_size_list_.push_back(input_row_ * sizeof(T));
  }

 private:
  int input_row_;
  int input_col_;
  int param_dim_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GPU_KERNEL_H_
