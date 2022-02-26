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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/mirror_pad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 2;
constexpr size_t kInputIndex1st = 1;
constexpr size_t kInputIndex2nd = 2;
constexpr size_t kInputIndex3rd = 3;
constexpr size_t kDimNeedPadBatch = 3;
constexpr size_t kDimNeedPadBatchAndChannel = 2;
constexpr size_t kInputXDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 2;
constexpr int kSymmetricCoef = 2;
constexpr size_t kIndexForMaxWidth = 3;
constexpr size_t kIndexForMaxHeight = 2;
constexpr size_t kMaxIndexOffset = 2;

template <typename T>
class MirrorPadFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  MirrorPadFwdGpuKernelMod()
      : num_input_(0),
        num_paddings_(0),
        mode_(0),
        is_null_input_(false),
        input_size_(1),
        output_size_(1),
        workspace_size_(0) {}
  ~MirrorPadFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    int64_t *paddings = GetDeviceAddress<int64_t>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t size = output_size_ / sizeof(T);
    int dim_offset = output_shape_.size() - kMaxIndexOffset;

    CalMirrorPad(size, input, input_shape_[0], input_shape_[kInputIndex1st], input_shape_[kInputIndex2nd],
                 input_shape_[kInputIndex3rd], output_shape_[dim_offset + 0], output_shape_[dim_offset + 1],
                 num_paddings_, paddings, mode_, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    string mode = GetValue<string>(prim->GetAttr("mode"));
    if (mode == "REFLECT") {
      mode_ = 0;  // reflected mirroring
    } else {
      mode_ = 1;  // symmetric mirroring
    }

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto padding_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input_x") ||
                     CHECK_SHAPE_NULL(padding_shape, kernel_name, "paddings") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    // shape adjustment -> from 2d/3d to 4d to standardize
    if (input_shape.size() == kDimNeedPadBatch) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, 1);  // batch padding
    } else if (input_shape.size() == kDimNeedPadBatchAndChannel) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, 2, 1);  // channel padding
    }
    if (input_shape.size() < kInputXDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input_x cannot be less than 4, but "
                        << "got the " << input_shape.size();
    }

    for (auto in_shape : input_shape) {
      input_size_ *= in_shape;
      input_shape_.push_back(in_shape);
    }
    num_input_ = input_size_;
    input_size_ *= sizeof(T);

    num_paddings_ = padding_shape[0];
    input_size_ += IntToSize(kSymmetricCoef) * num_paddings_ * sizeof(int64_t);

    output_size_ = sizeof(T);

    if (output_shape.size() < kOutputDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output cannot be less than 2, but "
                        << "got the " << output_shape.size();
    }
    for (auto x : output_shape) {
      output_size_ *= x;
      output_shape_.push_back(x);
    }

    int max_width = input_shape_[kIndexForMaxWidth];
    int max_height = input_shape_[kIndexForMaxHeight];
    // basic error check for padding value
    if (mode_ == 1) {  // symmetric
      max_width = max_width + (kSymmetricCoef * max_width);
      max_height = max_height + (kSymmetricCoef * max_height);
    } else {  // reflect
      max_width = max_width + (kSymmetricCoef * (max_width - 1));
      max_height = max_height + (kSymmetricCoef * (max_height - 1));
    }
    if (output_shape_[(output_shape_.size() - kMaxIndexOffset) + 0] > max_width ||
        output_shape_[(output_shape_.size() - kMaxIndexOffset) + 1] > max_width) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the output.shape[-1] and output.shape[-2] cannot be greater "
                        << "than input_x.shape[-1], but got output.shape: " << CONVERT_VECTOR_TO_STRING(output_shape_)
                        << ", input_x.shape: " << CONVERT_VECTOR_TO_STRING(input_shape_);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(num_input_ * sizeof(T));
    input_size_list_.push_back(kSymmetricCoef * num_paddings_ * sizeof(int64_t));  // for 64 bit int defined in API
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t num_input_;
  int num_paddings_;
  int mode_;
  bool is_null_input_;
  std::vector<int> input_shape_;   // dims of the input data
  std::vector<int> output_shape_;  // dims of the output data
  // default
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GPU_KERNEL_H_
