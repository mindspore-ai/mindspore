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

#include "ops/upsample_trilinear_3d.h"
#include "plugin/device/cpu/kernel/upsample_trilinear_3d_cpu_kernel.h"
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUpsampleTrilinear3DInputsNum = 1;
constexpr size_t kUpsampleTrilinear3DOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;
}  // namespace

bool UpsampleTrilinear3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  in_type_ = inputs.at(kIndex0)->GetDtype();
  auto kernel_ptr = std::make_shared<ops::UpsampleTrilinear3D>(base_operator->GetPrim());
  attr_scales_ = kernel_ptr->get_scales_attr();
  if (attr_scales_.empty()) {
    attr_scales_ = {0, 0, 0};
  }
  attr_align_corners_ = kernel_ptr->get_align_corners();
  return true;
}

int UpsampleTrilinear3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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

bool UpsampleTrilinear3DCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleTrilinear3DInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleTrilinear3DOutputNum, kernel_name_);

  bool res = false;
  switch (in_type_) {
    case kNumberTypeFloat16:
      res = LaunchKernel<float16, float>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      res = LaunchKernel<float, float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      res = LaunchKernel<double, double>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the dtype of 'x' should be float16, float32 or float64. But got "
                              << TypeIdLabel(in_type_);
  }
  return res;
}

template <typename T, typename S>
bool UpsampleTrilinear3DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  // treat batch and channels as one dimension
  int64_t channels = x_shape_[kIndex0] * x_shape_[kIndex1];
  int64_t input_depth = x_shape_[kIndex2];
  int64_t input_height = x_shape_[kIndex3];
  int64_t input_width = x_shape_[kIndex4];

  int64_t output_depth = y_shape_[kIndex2];
  int64_t output_height = y_shape_[kIndex3];
  int64_t output_width = y_shape_[kIndex4];

  MS_EXCEPTION_IF_CHECK_FAIL(channels > 0 && output_depth > 0 && output_height > 0 && output_width > 0,
                             "Invalid output shape.");

  auto x_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto y_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto cpy_ret = memcpy_s(y_ptr, outputs[kIndex0]->size, x_ptr, outputs[kIndex0]->size);
    if (cpy_ret != EOK) {
      MS_EXCEPTION(MemoryError) << "For " << kernel_name_ << ", memcpy_s to output failed.";
    }
    return true;
  }
  int64_t input_slice_size = input_depth * input_height * input_width;
  int64_t output_slice_size = output_depth * output_height * output_width;
  auto input_indexr_value = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
    return x_ptr[n * input_slice_size + d * input_height * input_width + h * input_width + w];
  };
  auto loop3d = [&](int64_t begin, int64_t end) {
    const S depth_scale =
      AreaPixelComputeScale<S>(input_depth, output_depth, attr_align_corners_, attr_scales_[kIndex0]);
    const S height_scale =
      AreaPixelComputeScale<S>(input_height, output_height, attr_align_corners_, attr_scales_[kIndex1]);
    const S width_scale =
      AreaPixelComputeScale<S>(input_width, output_width, attr_align_corners_, attr_scales_[kIndex2]);
    int64_t id0(0), id1(0), ih0(0), ih1(0), iw0(0), iw1(0);
    S d0lambda(0), d1lambda(0), h0lambda(0), h1lambda(0), w0lambda(0), w1lambda(0);
    for (int64_t n = begin; n < end; ++n) {
      for (int64_t od = 0; od < output_depth; ++od) {
        ComputeSourceIndexAndLambda(&id0, &id1, &d0lambda, &d1lambda, depth_scale, od, input_depth, output_depth,
                                    attr_align_corners_);
        for (int64_t oh = 0; oh < output_height; ++oh) {
          ComputeSourceIndexAndLambda(&ih0, &ih1, &h0lambda, &h1lambda, height_scale, oh, input_height, output_height,
                                      attr_align_corners_);
          for (int64_t ow = 0; ow < output_width; ++ow) {
            ComputeSourceIndexAndLambda(&iw0, &iw1, &w0lambda, &w1lambda, width_scale, ow, input_width, output_width,
                                        attr_align_corners_);
            auto i000 = static_cast<S>(input_indexr_value(n, id0, ih0, iw0));
            auto i001 = static_cast<S>(input_indexr_value(n, id0, ih0, iw1));
            auto i010 = static_cast<S>(input_indexr_value(n, id0, ih1, iw0));
            auto i011 = static_cast<S>(input_indexr_value(n, id0, ih1, iw1));
            auto i100 = static_cast<S>(input_indexr_value(n, id1, ih0, iw0));
            auto i101 = static_cast<S>(input_indexr_value(n, id1, ih0, iw1));
            auto i110 = static_cast<S>(input_indexr_value(n, id1, ih1, iw0));
            auto i111 = static_cast<S>(input_indexr_value(n, id1, ih1, iw1));
            double w000 = static_cast<double>(d0lambda) * h0lambda * w0lambda;
            double w001 = static_cast<double>(d0lambda) * h0lambda * w1lambda;
            double w010 = static_cast<double>(d0lambda) * h1lambda * w0lambda;
            double w011 = static_cast<double>(d0lambda) * h1lambda * w1lambda;
            double w100 = static_cast<double>(d1lambda) * h0lambda * w0lambda;
            double w101 = static_cast<double>(d1lambda) * h0lambda * w1lambda;
            double w110 = static_cast<double>(d1lambda) * h1lambda * w0lambda;
            double w111 = static_cast<double>(d1lambda) * h1lambda * w1lambda;
            y_ptr[n * output_slice_size + od * output_height * output_width + oh * output_width + ow] =
              static_cast<T>(w000 * i000 + w001 * i001 + w010 * i010 + w011 * i011 + w100 * i100 + w101 * i101 +
                             w110 * i110 + w111 * i111);
          }
        }
      }
    }
  };
  float block_size =
    SizeToLong(kGrainSize) > output_slice_size ? static_cast<float>(kGrainSize / output_slice_size) : 1.0;
  CPUKernelUtils::ParallelFor(loop3d, channels, block_size);
  return true;
}

std::vector<KernelAttr> UpsampleTrilinear3DCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
