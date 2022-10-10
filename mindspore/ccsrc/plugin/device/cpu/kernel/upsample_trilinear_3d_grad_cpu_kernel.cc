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

#include "plugin/device/cpu/kernel/upsample_trilinear_3d_grad_cpu_kernel.h"
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUpsampleTrilinear3DGradInputsNum = 1;
constexpr size_t kUpsampleTrilinear3DGradOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;
}  // namespace

void UpsampleTrilinear3DGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  in_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, kIndex0);
  // the input grad of backward process is the output of forward process
  output_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  input_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex0);
  if (output_shape_.size() != kDim5) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'grad_out' should be " << kDim5
                             << ", but got " << output_shape_.size();
  }
  attr_scales_ = common::AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, kAttrScales);
  if (attr_scales_.empty()) {
    attr_scales_ = {0, 0, 0};
  }
  attr_align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, kAttrAlignCorners);
}

bool UpsampleTrilinear3DGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleTrilinear3DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleTrilinear3DGradOutputNum, kernel_name_);
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
      break;
  }
  return res;
}

template <typename T, typename S>
bool UpsampleTrilinear3DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  // the input grad of backward process is the output of forward process
  auto grad_output_ptr = static_cast<T *>(inputs[kIndex0]->addr);
  const int64_t total = CPUKernelUtils::CalcElementNum(input_shape_);
  S *grad_input_ptr = nullptr;
  bool is_fp16 = std::is_same<T, float16>::value;
  // define for fp16
  std::vector<S> grad_input_copy(1);
  if (is_fp16) {
    grad_input_copy.resize(total, 0);
    grad_input_ptr = grad_input_copy.data();
  } else {
    grad_input_ptr = static_cast<S *>(outputs[kIndex0]->addr);
    std::fill_n(grad_input_ptr, total, S(0));
  }
  // treat nbatch and channels as one dimension
  int64_t channels = input_shape_[kIndex0] * input_shape_[kIndex1];
  int64_t input_depth = input_shape_[kIndex2];
  int64_t input_height = input_shape_[kIndex3];
  int64_t input_width = input_shape_[kIndex4];

  int64_t output_depth = output_shape_[kIndex2];
  int64_t output_height = output_shape_[kIndex3];
  int64_t output_width = output_shape_[kIndex4];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int64_t input_slice_size = input_depth * input_height * input_width;
  MS_EXCEPTION_IF_CHECK_FAIL(channels != 0 && output_depth != 0 && output_height != 0 && output_width != 0,
                             "Invalid output shape.");

  auto input_index = [=](int64_t c_idx, int64_t d_idx, int64_t h_idx, int64_t w_idx) {
    return c_idx * input_slice_size + d_idx * input_height * input_width + h_idx * input_width + w_idx;
  };
  auto loop3d = [&](int64_t begin, int64_t end) {
    const S depth_scale =
      AreaPixelComputeScale<S>(input_depth, output_depth, attr_align_corners_, attr_scales_[kIndex0]);
    const S height_scale =
      AreaPixelComputeScale<S>(input_height, output_height, attr_align_corners_, attr_scales_[kIndex1]);
    const S width_scale =
      AreaPixelComputeScale<S>(input_width, output_width, attr_align_corners_, attr_scales_[kIndex2]);

    int64_t id0{0}, id1{0}, ih0{0}, ih1{0}, iw0{0}, iw1{0};
    S d0lambda{0}, d1lambda{0}, h0lambda{0}, h1lambda{0}, w0lambda{0}, w1lambda{0};

    for (int64_t c_idx = begin; c_idx < end; ++c_idx) {
      for (int64_t od = 0; od < output_depth; ++od) {
        ComputeSourceIndexAndLambda(&id0, &id1, &d0lambda, &d1lambda, depth_scale, od, input_depth, output_depth,
                                    attr_align_corners_);

        for (int64_t oh = 0; oh < output_height; ++oh) {
          ComputeSourceIndexAndLambda(&ih0, &ih1, &h0lambda, &h1lambda, height_scale, oh, input_height, output_height,
                                      attr_align_corners_);

          for (int64_t ow = 0; ow < output_width; ++ow) {
            ComputeSourceIndexAndLambda(&iw0, &iw1, &w0lambda, &w1lambda, width_scale, ow, input_width, output_width,
                                        attr_align_corners_);

            auto grad_output_value = static_cast<S>(
              grad_output_ptr[c_idx * output_slice_size + od * output_height * output_width + oh * output_width + ow]);
            double w000 = static_cast<double>(d0lambda) * h0lambda * w0lambda;
            double w001 = static_cast<double>(d0lambda) * h0lambda * w1lambda;
            double w010 = static_cast<double>(d0lambda) * h1lambda * w0lambda;
            double w011 = static_cast<double>(d0lambda) * h1lambda * w1lambda;
            double w100 = static_cast<double>(d1lambda) * h0lambda * w0lambda;
            double w101 = static_cast<double>(d1lambda) * h0lambda * w1lambda;
            double w110 = static_cast<double>(d1lambda) * h1lambda * w0lambda;
            double w111 = static_cast<double>(d1lambda) * h1lambda * w1lambda;
            grad_input_ptr[input_index(c_idx, id0, ih0, iw0)] += static_cast<S>(w000 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id0, ih0, iw1)] += static_cast<S>(w001 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id0, ih1, iw0)] += static_cast<S>(w010 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id0, ih1, iw1)] += static_cast<S>(w011 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id1, ih0, iw0)] += static_cast<S>(w100 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id1, ih0, iw1)] += static_cast<S>(w101 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id1, ih1, iw0)] += static_cast<S>(w110 * grad_output_value);
            grad_input_ptr[input_index(c_idx, id1, ih1, iw1)] += static_cast<S>(w111 * grad_output_value);
          }
        }
      }
    }
  };
  float block_size = SizeToLong(kGrainSize) > output_slice_size
                       ? static_cast<float>(kGrainSize) / static_cast<float>(output_slice_size)
                       : 1.0;
  CPUKernelUtils::ParallelFor(loop3d, channels, block_size);
  // memcopy and cast for fp16
  if (is_fp16) {
    T *real_input_ptr = static_cast<T *>(outputs[kIndex0]->addr);
    for (int64_t idx = 0; idx < total; ++idx) {
      real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
    }
  }
  return true;
}

std::vector<KernelAttr> UpsampleTrilinear3DGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleTrilinear3DGrad, UpsampleTrilinear3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
