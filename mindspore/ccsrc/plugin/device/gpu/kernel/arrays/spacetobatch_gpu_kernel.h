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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/spacetobatch_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t PADDING_SHAPE_0 = 2;
constexpr size_t PADDING_SHAPE_1 = 2;

template <typename T>
class SpaceToBatchGpuKernelMod : public NativeGpuKernelMod {
 public:
  SpaceToBatchGpuKernelMod() { ResetResource(); }
  ~SpaceToBatchGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t size = input_size_ / sizeof(T);

    CalSpaceToBatch<T>(size, input, in_, ih_, iw_, ic_, on_, oh_, ow_, oc_, paddings_[0][0], paddings_[0][1],
                       paddings_[1][0], paddings_[1][1], block_size_, output,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckParam(kernel_node);
    input_size_ = sizeof(T);
    for (size_t idx = 0; idx < input_shape_.size(); ++idx) {
      input_size_ *= input_shape_[idx];
    }
    in_ = input_shape_[0];
    ic_ = input_shape_[1];
    ih_ = input_shape_[2];
    iw_ = input_shape_[3];

    on_ = in_ * block_size_ * block_size_;
    oc_ = ic_;
    oh_ = (ih_ + paddings_[0][0] + paddings_[0][1]) / block_size_;
    ow_ = (iw_ + paddings_[1][0] + paddings_[1][1]) / block_size_;
    output_size_ = on_ * oc_ * oh_ * ow_ * sizeof(T);
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    in_ = 0;
    ic_ = 0;
    ih_ = 0;
    iw_ = 0;
    on_ = 0;
    oc_ = 0;
    oh_ = 0;
    ow_ = 0;
    kernel_name_ = "SpaceToBatch";
    input_size_list_.clear();
    output_size_list_.clear();
    paddings_.clear();
    input_shape_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    block_size_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "block_size"));
    if (block_size_ < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 1, but got "
                        << block_size_;
    }
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }

    // check input_shape
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    if (input_shape.size() != SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to " << SHAPE_SIZE
                        << ", but got " << input_shape.size();
    }
    input_shape_.assign(input_shape.begin(), input_shape.end());
    // check paddings_
    paddings_ = GetAttr<std::vector<std::vector<int64_t>>>(kernel_node, "paddings");
    if (paddings_.size() != PADDING_SHAPE_0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'paddings' cannot be equal to " << PADDING_SHAPE_0
                        << ", but got " << paddings_.size();
    }
    if (paddings_[0].size() != PADDING_SHAPE_1 || paddings_[1].size() != PADDING_SHAPE_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'paddings' cannot be equal to " << PADDING_SHAPE_0
                        << ", but got " << paddings_.size();
    } else {
      for (size_t idx_i = 0; idx_i < PADDING_SHAPE_0; ++idx_i) {
        for (size_t idx_j = 0; idx_j < PADDING_SHAPE_1; ++idx_j) {
          if (paddings_[idx_i][idx_j] < 0) {
            MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the element of 'paddings' cannot be less than 0, "
                              << "but got paddings[" << idx_i << "][ " << idx_j << "]: " << paddings_[idx_i][idx_j];
          }
        }
        auto tmp_shape = input_shape[idx_i + PADDING_SHAPE_1] + paddings_[idx_i][0] + paddings_[idx_i][1];
        if ((tmp_shape % block_size_) != 0) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_
                            << "', padded shape should be divisible by block_size, , but got padded shape: "
                            << tmp_shape << ", block_size: " << block_size_;
        }
        if ((tmp_shape / block_size_) == 0) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', padded shape cannot be less than block_size"
                            << ", but got padded shape: " << tmp_shape << ", block_size: " << block_size_;
        }
      }
    }
    return true;
  }

  std::string kernel_name_;

  std::vector<std::vector<int64_t>> paddings_;
  std::vector<size_t> input_shape_;
  size_t block_size_;
  size_t input_size_;
  size_t output_size_;
  size_t in_;
  size_t ic_;
  size_t ih_;
  size_t iw_;
  size_t on_;
  size_t oc_;
  size_t oh_;
  size_t ow_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETOBATCH_KERNEL_H_
