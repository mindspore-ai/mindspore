/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_BOXES_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_BOXES_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNums = 4;
constexpr size_t kOutNum = 1;
constexpr size_t kGrads = 0;
constexpr size_t kGradsShapeLen = 4;
constexpr size_t kNumBoxes = 0;
constexpr size_t kHeight = 1;
constexpr size_t kWidth = 2;
constexpr size_t kDepth = 3;
constexpr size_t kBatch = 0;
constexpr size_t kImages = 1;
constexpr size_t kBoxes = 2;
constexpr size_t kImageShapeLen = 4;
constexpr size_t kCoordY1 = 0;
constexpr size_t kCoordX1 = 1;
constexpr size_t kCoordY2 = 2;
constexpr size_t kCoordX2 = 3;
constexpr size_t kBoxesShapeLen = 2;
constexpr size_t kCoordinateLen = 4;
constexpr size_t kBoxIndex = 3;
constexpr size_t kBoxIndexShapeLen = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kOutputShapeLen = 2;
constexpr float kNum = 0.5;
class CropAndResizeGradBoxesCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  CropAndResizeGradBoxesCpuKernelMod() = default;
  ~CropAndResizeGradBoxesCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  void OutputZeroing(const std::vector<AddressPtr> &outputs);

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using CropAndResizeGradBoxesFunc =
    std::function<bool(CropAndResizeGradBoxesCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, CropAndResizeGradBoxesFunc>> func_list_;
  CropAndResizeGradBoxesFunc kernel_func_;
  ShapeVector grads_shape_;
  ShapeVector image_shape_;
  ShapeVector boxes_shape_;
  ShapeVector box_in_shape_;
  ShapeVector output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_GRAD_BOXES_CPU_KERNEL_H_
