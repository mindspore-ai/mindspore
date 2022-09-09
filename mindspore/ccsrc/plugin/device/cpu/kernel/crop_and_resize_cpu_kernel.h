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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr int BILINEAR = 1;
constexpr int NEAREST = 2;
constexpr int BILINEAR_V2 = 3;
constexpr size_t INPUT_NUM = 4;
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t BOX_RANK = 2;
constexpr int64_t CROP_SIZE_LEN = 2;
constexpr size_t IMAGE_DIM = 4;
constexpr size_t IMAGE = 0;
constexpr size_t BOXES = 1;
constexpr size_t BOX_INDEX = 2;
constexpr size_t CROP_SIZE = 3;
constexpr size_t IMAGE_HEIGHT = 1;
constexpr size_t IMAGE_WEIGHT = 2;
class CropAndResizeCpuKernelMod : public NativeCpuKernelMod {
 public:
  CropAndResizeCpuKernelMod() = default;
  ~CropAndResizeCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitFunc(const CNodePtr &kernel_node);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  using CropAndResizeFunc = std::function<bool(CropAndResizeCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, CropAndResizeFunc>> func_list_;
  CropAndResizeFunc kernel_func_;
  int method_{1};
  float extrapolation_value_{0.0};
  int output_size_{0};
  int input_height_{0};
  int input_width_{0};
  int final_height_{0};
  int final_width_{0};
  int channel_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CROP_AND_RESIZE_CPU_KERNEL_H_
