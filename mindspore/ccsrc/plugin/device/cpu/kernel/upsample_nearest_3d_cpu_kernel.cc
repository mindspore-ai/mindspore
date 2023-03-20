/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/upsample_nearest_3d.h"
#include "plugin/device/cpu/kernel/upsample_nearest_3d_cpu_kernel.h"
#include <string>
#include <utility>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kUpsampleNearest3DInputsNum = 1;
constexpr auto kUpsampleNearest3DOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;

template <typename T>
inline T data_index_init(const T *offset) {
  return *offset;
}

template <typename T, typename... Args>
inline T data_index_init(T *offset, T *x, const T *X, Args &&... args) {
  auto off = data_index_init(offset, std::forward<Args>(args)...);
  *x = off % *X;
  return off / *X;
}

inline bool data_index_step() { return true; }

template <typename T, typename... Args>
inline bool data_index_step(T *x, const T *X, Args &&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == *X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
}
}  // namespace

bool UpsampleNearest3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  in_type_ = inputs.at(kIndex0)->GetDtype();
  auto kernel_ptr = std::make_shared<ops::UpsampleNearest3D>(base_operator->GetPrim());
  attr_scales_ = kernel_ptr->get_scales_attr();
  if (attr_scales_.empty()) {
    attr_scales_ = {0, 0, 0};
  }
  return true;
}

int UpsampleNearest3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  y_shape_ = outputs.at(kIndex0)->GetShapeVector();
  if (x_shape_.size() != kDim5) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'x' should be " << kDim5 << ", but got "
                             << x_shape_.size();
  }
  return KRET_OK;
}

bool UpsampleNearest3DCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleNearest3DInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleNearest3DOutputNum, kernel_name_);

  bool res = false;
  switch (in_type_) {
    case kNumberTypeFloat16:
      res = LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      res = LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      res = LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dtype of 'x' should be float16, float32, float64, but got " << TypeIdLabel(in_type_);
  }
  return res;
}

template <typename T>
bool UpsampleNearest3DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  int64_t channels = x_shape_[kIndex0] * x_shape_[kIndex1];
  int64_t input_depth = x_shape_[kIndex2];
  int64_t input_height = x_shape_[kIndex3];
  int64_t input_width = x_shape_[kIndex4];

  int64_t output_depth = y_shape_[kIndex2];
  int64_t output_height = y_shape_[kIndex3];
  int64_t output_width = y_shape_[kIndex4];
  int64_t input_slice_size = input_depth * input_height * input_width;

  auto x_ptr = static_cast<T *>(inputs[kIndex0]->addr);
  auto y_ptr = static_cast<T *>(outputs[kIndex0]->addr);
  (void)std::fill_n(y_ptr, CPUKernelUtils::CalcElementNum(y_shape_), T(0));
  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto cpy_ret = memcpy_s(y_ptr, outputs[kIndex0]->size, x_ptr, outputs[kIndex0]->size);
    if (cpy_ret != EOK) {
      MS_EXCEPTION(MemoryError) << "For " << kernel_name_ << ", memcpy_s to output failed.";
    }
    return true;
  }
  auto loop3d = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // data_index_init异常判断
    (void)data_index_init(&begin, &n, &channels, &od, &output_depth, &oh, &output_height, &ow, &output_width);
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t id = NearestIndex(od, input_depth, output_depth, static_cast<double>(attr_scales_[kIndex0]));
      int64_t ih = NearestIndex(oh, input_height, output_height, static_cast<double>(attr_scales_[kIndex1]));
      int64_t iw = NearestIndex(ow, input_width, output_width, static_cast<double>(attr_scales_[kIndex2]));
      y_ptr[idx] = x_ptr[n * input_slice_size + id * input_height * input_width + ih * input_width + iw];
      (void)data_index_step(&n, &channels, &od, &output_depth, &oh, &output_height, &ow, &output_width);
    }
  };
  CPUKernelUtils::ParallelFor(loop3d, static_cast<size_t>(CPUKernelUtils::CalcElementNum(y_shape_)),
                              static_cast<float>(kGrainSize));
  return true;
}
std::vector<KernelAttr> UpsampleNearest3DCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
