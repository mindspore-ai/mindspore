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

#include "plugin/device/cpu/kernel/upsample_nearest_3d_grad_cpu_kernel.h"
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kUpsampleNearest3DGradInputsNum = 1;
constexpr auto kUpsampleNearest3DGradOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;
}  // namespace

bool UpsampleNearest3DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  in_type_ = inputs.at(kIndex0)->GetDtype();
  auto kernel_ptr = std::make_shared<ops::UpsampleNearest3DGrad>(base_operator->GetPrim());
  attr_scales_ = kernel_ptr->get_scale_factors();
  if (attr_scales_.empty()) {
    attr_scales_ = {0, 0, 0};
  }
  return true;
}

int UpsampleNearest3DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_shape_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  input_shape_ = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  if (output_shape_.size() != kShape5dDims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'grads_output' should be " << kShape5dDims
                      << ", but got " << output_shape_.size();
  }
  return KRET_OK;
}

bool UpsampleNearest3DGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleNearest3DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleNearest3DGradOutputNum, kernel_name_);

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
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dtype of 'x' should be float16, float32, float64, but got " << TypeIdLabel(in_type_);
      break;
  }
  return res;
}

template <typename T, typename S>
bool UpsampleNearest3DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
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

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; ++c) {
      for (int64_t od = 0; od < output_depth; ++od) {
        int64_t id = NearestIndex(od, input_depth, output_depth, static_cast<double>(attr_scales_[kIndex0]));

        for (int64_t oh = 0; oh < output_height; ++oh) {
          int64_t ih = NearestIndex(oh, input_height, output_height, static_cast<double>(attr_scales_[kIndex1]));

          for (int64_t ow = 0; ow < output_width; ++ow) {
            int64_t iw = NearestIndex(ow, input_width, output_width, static_cast<double>(attr_scales_[kIndex2]));

            int64_t output_offset = c * output_slice_size + od * output_height * output_width + oh * output_width + ow;
            int64_t input_offset = c * input_slice_size + id * input_height * input_width + ih * input_width + iw;
            grad_input_ptr[input_offset] += static_cast<S>(grad_output_ptr[output_offset]);
          }
        }
      }
    }
  };
  CPUKernelUtils::ParallelFor(loop3d, channels, static_cast<float>(kGrainSize / output_slice_size));
  // memcopy and cast for fp16
  if (is_fp16) {
    T *real_input_ptr = static_cast<T *>(outputs[kIndex0]->addr);
    for (int64_t idx = 0; idx < total; ++idx) {
      real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
    }
  }
  return true;
}

std::vector<KernelAttr> UpsampleNearest3DGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleNearest3DGrad, UpsampleNearest3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
