/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GRAD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/layer_norm_grad_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class LayerNormGradGradGpuKernel : public GpuKernel {
 public:
  LayerNormGradGradGpuKernel() : input_row_(1), input_col_(1), param_dim_(1), input_size_(1) {}
  ~LayerNormGradGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto dy = GetDeviceAddress<T>(inputs, 1);
    auto var = GetDeviceAddress<T>(inputs, 2);
    auto mean = GetDeviceAddress<T>(inputs, 3);
    auto gamma = GetDeviceAddress<T>(inputs, 4);
    auto grad_dx = GetDeviceAddress<T>(inputs, 5);
    auto grad_dg = GetDeviceAddress<T>(inputs, 6);
    auto grad_db = GetDeviceAddress<T>(inputs, 7);
    auto d_x = GetDeviceAddress<T>(outputs, 0);
    auto d_dy = GetDeviceAddress<T>(outputs, 1);
    auto d_gamma = GetDeviceAddress<T>(outputs, 2);

    auto global_sum1 = GetDeviceAddress<T>(workspace, 0);
    auto global_sum2 = GetDeviceAddress<T>(workspace, 1);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(global_sum1, 0, input_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemsetAsync global_sum1 failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(global_sum2, 0, input_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemsetAsync global_sum2 failed");

    LayerNormGradGrad(input_row_, input_col_, param_dim_, global_sum1, global_sum2, epsilon_, dy, x, mean, var, gamma,
                      grad_dx, grad_dg, grad_db, d_dy, d_x, d_gamma, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    int begin_norm_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_norm_axis"));
    int begin_params_axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "begin_params_axis"));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (begin_norm_axis < 0) {
      begin_norm_axis += input_shape.size();
    }

    if (begin_params_axis < 0) {
      begin_params_axis += input_shape.size();
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

    epsilon_ = 1e-12;
    auto type_id = TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    if (std::strncmp(type_id, "kNumberTypeFloat16", std::strlen(type_id)) == 0) {
      epsilon_ = 1e-7;
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_ = input_row_ * input_col_ * sizeof(T);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(input_row_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(param_dim_ * sizeof(T));
    input_size_list_.push_back(param_dim_ * sizeof(T));

    output_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    output_size_list_.push_back(param_dim_ * sizeof(T));

    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    return;
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int input_row_;
  int input_col_;
  int param_dim_;
  int input_size_;
  T epsilon_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAYER_NORM_GRAD_GRAD_GPU_KERNEL_H_
