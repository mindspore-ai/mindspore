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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class SquaredDifferenceOpGpuKernel : public GpuKernel {
 public:
  SquaredDifferenceOpGpuKernel() { ResetResource(); }
  ~SquaredDifferenceOpGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *lhs = GetDeviceAddress<T>(inputs, 0);
    T *rhs = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (need_broadcast_) {
      BroadcastArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseArith(output_num_, op_type_, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto input_shape1 = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto input_shape2 = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape1) || CHECK_NULL_INPUT(input_shape2) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SquaredDifferenceGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    need_broadcast_ = IsBroadcast(input_shape1, input_shape2);
    if (need_broadcast_ && output_shape.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "Broadcast operation not support dim greater than " << MAX_DIMS;
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = output_shape[i];
      }
      output_num_ *= output_shape[i];
    }
    int lhs_offset = output_shape.size() - input_shape1.size();
    for (size_t j = 0; j < input_shape1.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = input_shape1[j];
        } else {
          auto index = j + lhs_offset;
          MS_LOG(EXCEPTION) << "Invalid input1 index: " << index;
        }
      }
      input1_num_ *= input_shape1[j];
    }
    int rhs_offset = output_shape.size() - input_shape2.size();
    for (size_t k = 0; k < input_shape2.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = input_shape2[k];
        } else {
          auto index = k + rhs_offset;
          MS_LOG(EXCEPTION) << "Invalid input2 index: " << index;
        }
      }
      input2_num_ *= input_shape2[k];
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    op_type_ = BROADCAST_TYPE_SQUARED_DIFFERENCE;
    need_broadcast_ = false;
    is_comp_op_ = false;
    is_null_input_ = false;
    input1_num_ = 1;
    input2_num_ = 1;
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
    input_size_list_.push_back(input1_num_ * sizeof(T));
    input_size_list_.push_back(input2_num_ * sizeof(T));
    output_size_list_.push_back(output_num_ * sizeof(T));
  }

 private:
  bool IsBroadcast(const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
    if (lhs.size() != rhs.size()) {
      return true;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i] != rhs[i]) {
        return true;
      }
    }
    return false;
  }

  BroadcastOpType op_type_;
  bool need_broadcast_;
  bool is_comp_op_;
  bool is_null_input_;
  size_t input1_num_;
  size_t input2_num_;
  size_t output_num_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARED_DIFFERENCE_GPU_KERNEL_H_
