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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_nd_functor_impl.cuh"

namespace mindspore {
namespace kernel {

static const std::map<std::string, ScatterNdFunctorType> kScatterNdFunctorTypeMap = {
  {"ScatterNdUpdate", SCATTER_ND_FUNC_UPDATE},
  {"ScatterNdAdd", SCATTER_ND_FUNC_ADD},
  {"ScatterNdSub", SCATTER_ND_FUNC_SUB},
};

template <typename T, typename S>
class ScatterNdFunctorKernelMod : public NativeGpuKernelMod {
 public:
  ScatterNdFunctorKernelMod() { ResetResource(); }
  ~ScatterNdFunctorKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *indices = GetDeviceAddress<S>(inputs, 1);
    T *updates = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    const size_t indices_len = sizeof(S) * out_strides_.size();
    S *indices_stride = GetDeviceAddress<S>(workspace, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(indices_stride, &out_strides_[0], indices_len, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    CalScatterNdFunctor(scatter_nd_functor_type_, unit_size_, num_units_, index_depth_, indices_stride, indices,
                        updates, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kScatterNdFunctorTypeMap.find(kernel_name);
    if (iter == kScatterNdFunctorTypeMap.end()) {
      MS_LOG(EXCEPTION)
        << "Only support these scatter functors: ScatterNdUpdate, ScatterNdAdd or ScatterNdSub currently, but got "
        << kernel_name;
    } else {
      scatter_nd_functor_type_ = iter->second;
    }
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto updates_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto index_depth = indices_shape.back();
    if (index_depth > input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the last dimension value of indices should be greater than "
                        << "the dimension of input , but got the dimension of input " << input_shape.size()
                        << ", got the last dimension value of indices " << index_depth;
    }
    if (indices_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of indices cannot be greater than 2, but got "
                        << indices_shape.size();
    }
    if (updates_shape.size() != indices_shape.size() - 1 + input_shape.size() - index_depth) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the dimension of updates, indices, shape should be consistent.";
    }
    for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
      if (updates_shape[i] != indices_shape[i]) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name << ", value of " << i
                          << "th dimension of indices is not equal to that update";
      }
    }

    indices_size_ = 1;
    for (size_t i = 0; i < indices_shape.size(); i++) {
      indices_size_ *= indices_shape[i];
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    updates_size_ = 1;
    for (size_t i = 0; i < updates_shape.size(); i++) {
      updates_size_ *= updates_shape[i];
    }

    index_depth_ = SizeToInt(index_depth);
    unit_size_ = 1;
    for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
      unit_size_ *= SizeToInt(updates_shape[i]);
    }
    num_units_ = 1;
    num_units_ *= updates_shape[indices_shape.size() - 2];
    for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
      num_units_ *= updates_shape[i];
    }
    int out_stride = 1;
    out_strides_.push_back(out_stride);
    for (int i = index_depth_ - 2; i >= 0; i--) {
      out_stride *= input_shape[i + 1];
      out_strides_.push_back(out_stride);
    }
    reverse(out_strides_.begin(), out_strides_.end());
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    indices_size_ = 0;
    updates_size_ = 0;
    unit_size_ = 0;
    num_units_ = 0;
    index_depth_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(indices_size_ * sizeof(S));
    input_size_list_.push_back(updates_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
    workspace_size_list_.push_back(sizeof(S) * out_strides_.size());
  }

 private:
  ScatterNdFunctorType scatter_nd_functor_type_;
  size_t input_size_;
  size_t indices_size_;
  size_t updates_size_;

  size_t unit_size_;
  size_t num_units_;
  size_t index_depth_;
  std::vector<S> out_strides_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ND_FUNCTOR_GPU_KERNEL_H_
