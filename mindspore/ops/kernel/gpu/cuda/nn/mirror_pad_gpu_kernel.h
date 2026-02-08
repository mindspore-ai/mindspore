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
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/mirror_pad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 2;
constexpr size_t kInputIndex1st = 1;
constexpr size_t kInputIndex2nd = 2;
constexpr size_t kInputIndex3rd = 3;
constexpr size_t kDimMin = 2;
constexpr size_t kDimMax = 4;
constexpr size_t kDimNeedPadBatch = 3;
constexpr size_t kDimNeedPadBatchAndChannel = 2;
constexpr size_t kInputXDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 2;
constexpr int kSymmetricCoef = 2;
constexpr size_t kIndexForMaxWidth = 3;
constexpr size_t kIndexForMaxHeight = 2;
constexpr size_t kMaxIndexOffset = 2;

class MirrorPadGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<MirrorPadGpuKernelMod> {
 public:
  MirrorPadGpuKernelMod()
      : num_input_(0),
        num_paddings_(0),
        mode_(0),
        is_null_input_(false),
        input_size_(1),
        output_size_(1),
        workspace_size_(0) {}
  ~MirrorPadGpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs) {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    int64_t *paddings = GetDeviceAddress<int64_t>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t size = output_size_ / sizeof(T);
    int dim_offset = output_shape_.size() - kMaxIndexOffset;

    auto status =
      CalMirrorPad(size, input, input_shape_[0], input_shape_[kInputIndex1st], input_shape_[kInputIndex2nd],
                   input_shape_[kInputIndex3rd], output_shape_[dim_offset + 0], output_shape_[dim_offset + 1],
                   num_paddings_, paddings, mode_, output, reinterpret_cast<cudaStream_t>(stream_ptr_));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
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
  size_t in_type_size_{1};
  size_t out_type_size_{1};
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GPU_KERNEL_H_
