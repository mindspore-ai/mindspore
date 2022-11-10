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

#include <algorithm>
#include <numeric>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include "plugin/device/cpu/kernel/adaptive_max_pool_2d_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
#define F64 kNumberTypeFloat64
#define F32 kNumberTypeFloat32
#define F16 kNumberTypeFloat16
#define I32 kNumberTypeInt32
#define I64 kNumberTypeInt64
constexpr size_t k4D = 4;
constexpr size_t k3D = 3;
constexpr size_t kInputsNum = 3;
constexpr size_t kOutputsNum = 1;
constexpr int64_t kOne = 1;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;

template <typename SCALAR_T, typename INDICES_T>
struct AdaptiveCalcArgs {
  SCALAR_T *input_grad_data = nullptr;
  SCALAR_T *output_grad_data = nullptr;
  INDICES_T *indices_data = nullptr;
  int64_t in_size_b = 0;
  int64_t in_size_d = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
};
}  // namespace

bool AdaptiveMaxPool2DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdaptiveMaxPool2DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_y_grad_shape = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  input_x_shape = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
  input_argmax_shape = inputs.at(kIndex2)->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename SCALAR_T, typename INDICES_T>
CTask AdaptiveMaxPool2DGradOutFrame(const AdaptiveCalcArgs<SCALAR_T, INDICES_T> &args) {
  auto shard_frame = [&args](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      SCALAR_T *grad_input_p_d = args.input_grad_data + d * args.in_size_h * args.in_size_w;
      SCALAR_T *grad_output_p_d = args.output_grad_data + d * args.out_size_h * args.out_size_w;
      INDICES_T *ind_p_d = args.indices_data + d * args.out_size_h * args.out_size_w;
      /* calculate max points */
      int64_t oh, ow;
      for (oh = 0; oh < args.out_size_h; oh++) {
        for (ow = 0; ow < args.out_size_w; ow++) {
          /* retrieve position of max */
          INDICES_T maxp = ind_p_d[oh * args.out_size_w + ow];

          grad_input_p_d[maxp] += grad_output_p_d[oh * args.out_size_w + ow];
        }
      }
    }
  };
  return shard_frame;
}

template <typename SCALAR_T, typename INDICES_T>
bool AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  const size_t input_x_dims = input_x_shape.size();
  int64_t dim_w = 2;
  int64_t dim_h = 1;
  int64_t size_b = 1;
  if (input_x_dims == k4D) {
    size_b = input_x_shape[0];
    dim_w++;
    dim_h++;
  }
  AdaptiveCalcArgs<SCALAR_T, INDICES_T> args;
  args.in_size_b = size_b;
  args.in_size_d = input_x_shape[dim_h - kOne];
  args.in_size_h = input_x_shape[dim_h];
  args.in_size_w = input_x_shape[dim_w];
  args.out_size_h = input_y_grad_shape[dim_h];
  args.out_size_w = input_y_grad_shape[dim_w];

  auto input_grad_data_ptr_ret = static_cast<SCALAR_T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_grad_data_ptr_ret);
  int64_t output_num = std::accumulate(input_x_shape.cbegin(), input_x_shape.cend(), 1, std::multiplies<int64_t>{});
  std::unique_ptr<SCALAR_T[]> input_grad_data_ptr = std::make_unique<SCALAR_T[]>(output_num);
  std::fill_n(input_grad_data_ptr.get(), output_num, static_cast<SCALAR_T>(0));
  auto output_grad_data_ptr = static_cast<SCALAR_T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_grad_data_ptr);
  auto indices_data_ptr = static_cast<INDICES_T *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(indices_data_ptr);
  // resize output
  if (input_x_dims == k3D) {
    args.input_grad_data = input_grad_data_ptr.get();
    args.output_grad_data = output_grad_data_ptr;
    args.indices_data = indices_data_ptr;
    auto shard_frame = AdaptiveMaxPool2DGradOutFrame<SCALAR_T, INDICES_T>(args);
    ParallelLaunchAutoSearch(shard_frame, args.in_size_d, this, &parallel_search_info_);
  } else {
    auto shard_template = [&args, &input_grad_data_ptr, &output_grad_data_ptr, &indices_data_ptr, this](int64_t start,
                                                                                                        int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T, INDICES_T> sub_args = args;
        sub_args.input_grad_data = input_grad_data_ptr.get() + b * args.in_size_d * args.in_size_h * args.in_size_w;
        sub_args.output_grad_data = output_grad_data_ptr + b * args.in_size_d * args.out_size_h * args.out_size_w;
        sub_args.indices_data = indices_data_ptr + b * args.in_size_d * args.out_size_h * args.out_size_w;
        auto shard_frame = AdaptiveMaxPool2DGradOutFrame<SCALAR_T, INDICES_T>(sub_args);
        ParallelLaunchAutoSearch(shard_frame, sub_args.in_size_d, this, &parallel_search_info_);
      }
    };
    ParallelLaunchAutoSearch(shard_template, args.in_size_b, this, &parallel_search_info_);
  }
  for (int64_t i = 0; i < output_num; i++) {
    input_grad_data_ptr_ret[i] = static_cast<SCALAR_T>(input_grad_data_ptr[i]);
  }
  return true;
}

std::vector<std::pair<KernelAttr, AdaptiveMaxPool2DGradCpuKernelMod::AdaptiveMaxPool2DGradLaunchFunc>>
  AdaptiveMaxPool2DGradCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I32).AddOutputAttr(F16),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I32).AddOutputAttr(F32),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I32).AddOutputAttr(F64),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I64).AddOutputAttr(F16),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I64).AddOutputAttr(F32),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I64).AddOutputAttr(F64),
     &AdaptiveMaxPool2DGradCpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> AdaptiveMaxPool2DGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, AdaptiveMaxPool2DGradCpuKernelMod::AdaptiveMaxPool2DGradLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool2DGrad, AdaptiveMaxPool2DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
