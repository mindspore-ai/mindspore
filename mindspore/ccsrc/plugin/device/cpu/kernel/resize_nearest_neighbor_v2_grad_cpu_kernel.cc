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

#include <string>
#include "plugin/device/cpu/kernel/resize_nearest_neighbor_v2_grad_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/grad/resize_nearest_neighbor_v2_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborV2GradInputsNum = 2;
constexpr size_t kResizeNearestNeighborV2GradOutputNum = 1;
}  // namespace

bool ResizeNearestNeighborV2GradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                   const std::vector<KernelTensorPtr> &inputs,
                                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto op_prim = std::dynamic_pointer_cast<ops::ResizeNearestNeighborV2Grad>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  align_corners_ = op_prim->get_align_corners();
  half_pixel_centers_ = op_prim->get_half_pixel_centers();

  std::string data_format = op_prim->get_data_format();
  if (data_format.compare(kOpFormat_NCHW) == 0) {
    dim_idx_map_ = {{'N', kIndex0}, {'C', kIndex1}, {'H', kIndex2}, {'W', kIndex3}};
  } else if (data_format.compare(kOpFormat_NHWC) == 0) {
    dim_idx_map_ = {{'N', kIndex0}, {'H', kIndex1}, {'W', kIndex2}, {'C', kIndex3}};
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the attr of 'data_format' only support [" << kOpFormat_NCHW << ", "
                  << kOpFormat_NHWC << "].";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int ResizeNearestNeighborV2GradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs,
                                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  y_type_ = outputs[kIndex0]->GetDtype();
  y_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  grads_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  auto size_shape = inputs[kIndex1]->GetShapeVector();
  if (grads_shape_.size() != kShape4dDims && !IsDynamicRank(grads_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be " << kShape4dDims << ", but got "
                  << grads_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (size_shape.size() != kShape1dDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'size' should be " << kShape1dDims << ", but got "
                  << size_shape.size();
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool ResizeNearestNeighborV2GradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborV2GradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborV2GradOutputNum, kernel_name_);
  bool res = false;
  switch (y_type_) {
    case kNumberTypeUInt8:
      res = LaunchKernel<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      res = LaunchKernel<uint16_t>(inputs, outputs);
      break;
    case kNumberTypeInt8:
      res = LaunchKernel<int8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      res = LaunchKernel<int16_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      res = LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      res = LaunchKernel<int64_t>(inputs, outputs);
      break;
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
      MS_EXCEPTION(TypeError)
        << "For '" << kernel_name_
        << "', the dtype of 'x' should be float16, float32, float64, int32, int64, int16, int8, uint16 or uin8 but got "
        << TypeIdLabel(y_type_);
      break;
  }
  return res;
}
template <typename T>
bool ResizeNearestNeighborV2GradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                           const std::vector<kernel::AddressPtr> &outputs) {
  const int64_t batch_size = grads_shape_[dim_idx_map_['N']];
  const int64_t in_height = grads_shape_[dim_idx_map_['H']];
  const int64_t in_width = grads_shape_[dim_idx_map_['W']];
  const int64_t channels = grads_shape_[dim_idx_map_['C']];
  const int64_t out_height = y_shape_[dim_idx_map_['H']];
  const int64_t out_width = y_shape_[dim_idx_map_['W']];

  const float height_scale = Scaling(out_height, in_height, align_corners_);
  const float width_scale = Scaling(out_width, in_width, align_corners_);

  auto grads_4d = EigenTensor(grads_shape_, inputs[kIndex0]->addr).tensor<T, kDim4>();
  auto y_4d = EigenTensor(y_shape_, outputs[kIndex0]->addr).tensor<T, kDim4>();
  y_4d.setZero();

  for (int64_t y = 0; y < in_height; ++y) {
    int64_t out_y =
      std::min((align_corners_) ? static_cast<int64_t>(roundf(Scaler(y, height_scale, half_pixel_centers_)))
                                : static_cast<int64_t>(floorf(Scaler(y, height_scale, half_pixel_centers_))),
               out_height - 1);
    for (int64_t x = 0; x < in_width; ++x) {
      int64_t out_x =
        std::min((align_corners_) ? static_cast<int64_t>(roundf(Scaler(x, width_scale, half_pixel_centers_)))
                                  : static_cast<int64_t>(floorf(Scaler(x, width_scale, half_pixel_centers_))),
                 out_width - 1);
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
          // data_format = NHWC
          if (dim_idx_map_['C'] == kIndex3) {
            y_4d(b, out_y, out_x, c) += grads_4d(b, y, x, c);
          } else {
            // data_format = NCHW
            y_4d(b, c, out_y, out_x) += grads_4d(b, c, y, x);
          }
        }
      }
    }
  }
  return true;
}

std::vector<KernelAttr> ResizeNearestNeighborV2GradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
