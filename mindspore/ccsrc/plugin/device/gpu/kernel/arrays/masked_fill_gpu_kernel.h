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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MASKED_FILL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MASKED_FILL_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/masked_fill_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class MaskedFillGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  MaskedFillGpuKernelMod() { ResetResource(); }
  ~MaskedFillGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    bool *mask_addr = GetDeviceAddress<bool>(inputs, 1);
    T *value = GetDeviceAddress<T>(inputs, 2);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T value_cut;

    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(&value_cut, value, sizeof(T), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync MaskedFill value failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "MaskedFill cudaStreamSynchronized failed");

    if (need_broadcast_) {
      BroadcastMaskedFill(lhs_shape_, rhs_shape_, output_shape_, input_addr, mask_addr, value_cut, output_addr,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseMaskedFill(input_num_, input_addr, mask_addr, value_cut, output_addr,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto mask_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") ||
                     CHECK_SHAPE_NULL(mask_shape, kernel_name_, "mask") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(input_shape, mask_shape);
    if (need_broadcast_ && input_shape.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                        << ", but got " << input_shape.size();
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        if (i < MAX_DIMS) {
          output_shape_[i] = output_shape[i];
        } else {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of output should be less than " << MAX_DIMS
                            << ", but got " << i;
        }
      }
      output_num_ *= output_shape[i];
    }
    int lhs_offset = output_shape.size() - input_shape.size();
    for (size_t j = 0; j < input_shape.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = input_shape[j];
        } else {
          auto index = j + lhs_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name_
                            << "', the index of input cannot be less than 0 and greater than " << MAX_DIMS
                            << ", but got " << index;
        }
      }
      input_num_ *= input_shape[j];
    }
    int rhs_offset = output_shape.size() - mask_shape.size();
    for (size_t k = 0; k < mask_shape.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = mask_shape[k];
        } else {
          auto index = k + rhs_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of mask cannot be less than 0 and greater than "
                            << MAX_DIMS << ", but got " << index;
        }
      }
      mask_num_ *= mask_shape[k];
    }

    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    need_broadcast_ = false;
    is_null_input_ = false;
    input_num_ = 1;
    mask_num_ = 1;
    output_num_ = 1;
    lhs_shape_.clear();
    rhs_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { return; }
  void InitSizeLists() override {
    input_size_list_.push_back(input_num_ * sizeof(T));
    input_size_list_.push_back(mask_num_ * sizeof(bool));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(output_num_ * sizeof(T));
  }

 private:
  bool need_broadcast_;
  bool is_null_input_;
  size_t input_num_;
  size_t mask_num_;
  size_t output_num_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MASKED_FILL_GPU_KERNEL_H_
