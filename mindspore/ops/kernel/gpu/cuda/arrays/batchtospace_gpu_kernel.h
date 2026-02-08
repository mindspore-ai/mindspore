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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/batchtospace_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t CROPS_SHAPE_0 = 2;
constexpr size_t CROPS_SHAPE_1 = 2;
template <typename T>
class BatchToSpaceGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchToSpaceGpuKernelMod() {
    in_ = 0;
    ic_ = 0;
    ih_ = 0;
    iw_ = 0;
    on_ = 0;
    oc_ = 0;
    oh_ = 0;
    ow_ = 0;
    kernel_name_ = "BatchToSpace";
    crops_.clear();
    output_size_list_.clear();
    input_shape_.clear();
  }
  ~BatchToSpaceGpuKernelMod() = default;
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t size = output_size_list_[0] / sizeof(T);

    auto status =
      CalBatchToSpace<T>(size, input, in_, ih_, iw_, ic_, on_, oh_, ow_, oc_, crops_[0][0], crops_[0][1], crops_[1][0],
                         crops_[1][1], block_size_, output, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    // wait for primitive unified between lite and cloud.
    block_size_ = static_cast<size_t>(GetValue<int64_t>(primitive_->GetAttr("block_size")));
    if (block_size_ < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 1, but got "
                        << block_size_;
    }
    // check crops
    crops_ = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("crops"));
    if (crops_.size() != CROPS_SHAPE_0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'crops' must be " << CROPS_SHAPE_0 << ", but got "
                        << crops_.size();
    }
    if (crops_[0].size() != CROPS_SHAPE_1 || crops_[1].size() != CROPS_SHAPE_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of element of 'crops' must be " << CROPS_SHAPE_1
                        << ", but got the size of crops[0]: " << crops_[0].size()
                        << ", the size of crops[1]: " << crops_[1].size();
    }
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), 1, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    // check input_shape
    auto input_shape = inputs[0]->GetShapeVector();
    if (input_shape.size() != SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 4, but got "
                        << input_shape.size();
    }
    if ((input_shape[0] % (block_size_ * block_size_)) != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', input_shape[0] must be divisible by product of block_shape, but got input_shape[0]: "
                        << input_shape[0] << ", block_shape: " << block_size_;
    }
    for (size_t idx = 0; idx < SHAPE_SIZE; ++idx) {
      if (input_shape[idx] < 1) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the element of shape of input cannot be less than 1, but got " << input_shape;
      }
    }
    input_shape_.assign(input_shape.begin(), input_shape.end());
    for (size_t idx_i = 0; idx_i < CROPS_SHAPE_0; ++idx_i) {
      for (size_t idx_j = 0; idx_j < CROPS_SHAPE_1; ++idx_j) {
        if (crops_[idx_i][idx_j] < 0) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_
                            << "', the element of 'crops' must be greater than or equal to 0, but got crops[" << idx_i
                            << "][" << idx_j << "]: " << crops_[idx_i][idx_j];
        }
      }
      auto tmp_shape = input_shape[idx_i + CROPS_SHAPE_1] * block_size_ - crops_[idx_i][0] - crops_[idx_i][1];
      if (tmp_shape <= 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the element of shape of output must be greater than 0, but got " << tmp_shape;
      }
    }
    constexpr int IDX_2 = 2;
    constexpr int IDX_3 = 3;
    in_ = static_cast<size_t>(input_shape_[0]);
    ic_ = static_cast<size_t>(input_shape_[1]);
    ih_ = static_cast<size_t>(input_shape_[IDX_2]);
    iw_ = static_cast<size_t>(input_shape_[IDX_3]);

    on_ = in_ / (block_size_ * block_size_);
    oc_ = ic_;
    oh_ = ih_ * block_size_ - crops_[0][0] - crops_[0][1];
    ow_ = iw_ * block_size_ - crops_[1][0] - crops_[1][1];
    return static_cast<int>(KRET_OK);
  }

 private:
  std::vector<std::vector<int64_t>> crops_;
  std::vector<int64_t> input_shape_;
  size_t block_size_;
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
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_
