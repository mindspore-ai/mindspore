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

#include "plugin/device/cpu/kernel/resize_nearest_neighbor_v2_cpu_kernel.h"
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborV2InputsNum = 2;
constexpr size_t kResizeNearestNeighborV2OutputNum = 1;
}  // namespace

void ResizeNearestNeighborV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  cnode_ptr_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  y_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, kIndex0);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  y_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex0);
  auto size_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  if (x_shape_.size() != kShape4dDims && !IsDynamicRank(x_shape_)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'x' should be " << kShape4dDims
                             << ", but got " << x_shape_.size();
  }
  if (size_shape.size() != kShape1dDims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'size' should be " << kShape1dDims
                             << ", but got " << size_shape.size();
  }
  align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, kAttrAlignCorners);
  half_pixel_centers_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, kAttrHalfPixelCenters);
  std::string data_format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFormat);

  if (data_format.compare(kOpFormat_NCHW) == 0) {
    dim_idx_map_ = {{'N', kIndex0}, {'C', kIndex1}, {'H', kIndex2}, {'W', kIndex3}};
  } else if (data_format.compare(kOpFormat_NHWC) == 0) {
    dim_idx_map_ = {{'N', kIndex0}, {'H', kIndex1}, {'W', kIndex2}, {'C', kIndex3}};
  } else {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the attr of 'data_format' only support ["
                             << kOpFormat_NCHW << ", " << kOpFormat_NHWC << "].";
  }
}

bool ResizeNearestNeighborV2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborV2OutputNum, kernel_name_);
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
  }
  return res;
}

template <typename T>
bool ResizeNearestNeighborV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  const int64_t batch_size = x_shape_[dim_idx_map_['N']];
  const int64_t in_height = x_shape_[dim_idx_map_['H']];
  const int64_t in_width = x_shape_[dim_idx_map_['W']];
  const int64_t channels = x_shape_[dim_idx_map_['C']];
  const int64_t out_height = y_shape_[dim_idx_map_['H']];
  const int64_t out_width = y_shape_[dim_idx_map_['W']];

  const float height_scale = Scaling(static_cast<size_t>(in_height), static_cast<size_t>(out_height), align_corners_);
  const float width_scale = Scaling(static_cast<size_t>(in_width), static_cast<size_t>(out_width), align_corners_);

  auto x_4d = EigenTensor(x_shape_, inputs[kIndex0]->addr).tensor<T, kDim4>();
  auto y_4d = EigenTensor(y_shape_, outputs[kIndex0]->addr).tensor<T, kDim4>();
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t y = 0; y < out_height; ++y) {
      int64_t in_y =
        std::min((align_corners_)
                   ? static_cast<int64_t>(roundf(Scaler(static_cast<size_t>(y), height_scale, half_pixel_centers_)))
                   : static_cast<int64_t>(floorf(Scaler(static_cast<size_t>(y), height_scale, half_pixel_centers_))),
                 in_height - 1);
      if (half_pixel_centers_) {
        in_y = std::max(static_cast<int64_t>(0), in_y);
      }
      for (int64_t x = 0; x < out_width; ++x) {
        int64_t in_x =
          std::min((align_corners_)
                     ? static_cast<int64_t>(roundf(Scaler(static_cast<size_t>(x), width_scale, half_pixel_centers_)))
                     : static_cast<int64_t>(floorf(Scaler(static_cast<size_t>(x), width_scale, half_pixel_centers_))),
                   in_width - 1);
        if (half_pixel_centers_) {
          in_x = std::max(static_cast<int64_t>(0), in_x);
        }
        // data_format = NHWC
        if (dim_idx_map_['C'] == kIndex3) {
          (void)std::copy_n(&x_4d(b, in_y, in_x, 0), channels, &y_4d(b, y, x, 0));
        } else {
          // data_format = NCHW
          for (int64_t c = 0; c < channels; ++c) {
            y_4d(b, c, y, x) = x_4d(b, c, in_y, in_x);
          }
        }
      }
    }
  }
  return true;
}

std::vector<KernelAttr> ResizeNearestNeighborV2CpuKernelMod::GetOpSupport() {
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
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighborV2, ResizeNearestNeighborV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
