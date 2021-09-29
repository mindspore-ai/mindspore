/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/layer_norm_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class LayerNormGradGpuKernel : public GpuKernel {
 public:
  LayerNormGradGpuKernel() : input_row_(1), input_col_(1), param_dim_(1), is_null_input_(false) {}
  ~LayerNormGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto dy = GetDeviceAddress<T>(inputs, 1);
    auto var = GetDeviceAddress<T>(inputs, 2);
    auto mean = GetDeviceAddress<T>(inputs, 3);
    auto gamma = GetDeviceAddress<T>(inputs, 4);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    auto dg = GetDeviceAddress<T>(outputs, 1);
    auto db = GetDeviceAddress<T>(outputs, 2);

    const T epsilon = 10e-12;
    LayerNormGrad(input_row_, input_col_, param_dim_, epsilon, dy, x, mean, var, gamma, dx, dg, db,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    int begin_norm_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_norm_axis"));
    int begin_params_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_params_axis"));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'LayerNormGradGpuKernel', input is null.";
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
      MS_LOG(EXCEPTION) << "For 'LayerNormGradGpuKernel', begin_norm_axis should be less than or equal to "
                        << "the rank of input, but got begin_norm_axis: " << IntToSize(begin_norm_axis)
                        << ", rank of input: " << input_shape.size();
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
    input_size_list_.push_back(input_row_ * input_col_ * sizeof(T));
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));

    output_size_list_.push_back(input_row_ * input_col_ * sizeof(T));
    output_size_list_.push_back(param_dim_ * sizeof(T));
    output_size_list_.push_back(param_dim_ * sizeof(T));
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int input_row_;
  int input_col_;
  int param_dim_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GPU_KERNEL_H_
