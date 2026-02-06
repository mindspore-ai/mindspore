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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <bitset>
#include <algorithm>
#include <map>
#include <utility>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda/arrays/strided_slice_gpu_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kStridedSliceMaxDims = 8;
class StridedSliceGradGpuKernelMod : public NativeGpuKernelMod, public StridedSliceGpuCommon {
 public:
  StridedSliceGradGpuKernelMod() = default;
  ~StridedSliceGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, outputs);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport();
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override {
    return {kShapexIndex_, kBeginIndex_, kEndIndex_, kStrideIndex_};
  }

 protected:
  bool is_null_input_{false};
  static constexpr size_t kShapexIndex_{1};
  static constexpr size_t kBeginIndex_{2};
  static constexpr size_t kEndIndex_{3};
  static constexpr size_t kStrideIndex_{4};
  std::vector<int64_t> shapex_;

 private:
  using StridedSliceGradLaunchFunc =
    std::function<bool(StridedSliceGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  template <typename T, typename S = int64_t>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void FillEmptyDims(const std::string &kernel_name, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                     std::vector<int64_t> *stride, ShapeVector *input_shape);
  void ComputeBeginMask(std::vector<int64_t> *begin, const std::vector<int64_t> &stride, const ShapeVector &input_shape,
                        const PrimitivePtr &primitive_);
  void ComputeEndMask(std::vector<int64_t> *end, const std::vector<int64_t> &stride, const ShapeVector &input_shape,
                      const PrimitivePtr &primitive_);
  void ComputeEllipsisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                           const ShapeVector &input_shape, const PrimitivePtr &primitive_);
  void ComputNewAxisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                         const ShapeVector &input_shape, const PrimitivePtr &primitive_);
  void ComputeShrinkAxisMask(const std::vector<int64_t> &begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                             const PrimitivePtr &primitive_);
  void *cuda_stream_{nullptr};
  StridedSliceGradLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, StridedSliceGradLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
