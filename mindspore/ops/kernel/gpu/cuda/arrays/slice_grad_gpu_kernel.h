/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <map>
#include <memory>
#include "primitive/op_name.h"
#include "mindspore/ops/infer/grad/slice_grad.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/slice_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_class/slice_grad_helper.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSliceGradDefaultInputShapeSize = 4;
constexpr size_t kSliceGradMaxInputShapeSize = 7;
constexpr size_t DynamicInputNum = 4;
constexpr size_t kBeginIndex_ = 2;
constexpr size_t kSizeIndex_ = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim7 = 7;

class SliceGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SliceGradGpuKernelMod();
  ~SliceGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kBeginIndex_, kSizeIndex_}; }

 private:
  void ProccessAttr(const std::vector<KernelTensor *> &inputs);
  void CalcBeginAndSize(const mindspore::Format &data_format, size_t dim = kDim4);
  void CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::vector<int64_t> strides_;
  ShapeVector input_shape_;
  ShapeVector dy_shape_;

  std::string kernel_name_;
  std::shared_ptr<cukernel::SliceGradAttr> attr_ptr_{nullptr};

  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
