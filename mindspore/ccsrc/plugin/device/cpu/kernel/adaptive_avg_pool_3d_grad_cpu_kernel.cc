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

#include <cmath>
#include <memory>
#include "plugin/device/cpu/kernel/adaptive_avg_pool_3d_grad_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t k5D = 5;
constexpr size_t k4D = 4;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
constexpr int64_t kIdxR4th = -4;
constexpr int64_t kIdxR3rd = -3;
constexpr int64_t kIdxR2nd = -2;
constexpr int64_t kIdxR1st = -1;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  double *input_data = nullptr;
  SCALAR_T *output_data = nullptr;
  int64_t size_d = 0;
  int64_t in_size_t = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
  int64_t out_size_t = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
};

inline int64_t StartIndex(int64_t offset, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(std::floor(static_cast<float>(offset * in_size) / out_size));
}

inline int64_t EndIndex(int64_t offset, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(std::ceil(static_cast<float>((offset + 1) * in_size) / out_size));
}
}  // namespace

bool AdaptiveAvgPool3DGradCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
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

int AdaptiveAvgPool3DGradCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  grad_output_dim_sizes_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  size_t input_1_shape_size = grad_output_dim_sizes_.size();
  if (input_1_shape_size != k4D && input_1_shape_size != k5D) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimensions of input should be 4 or 5, but got "
                      << input_1_shape_size;
  }
  orig_input_shape_dim_sizes_ = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
  size_t input_shape_size = orig_input_shape_dim_sizes_.size();
  if (input_shape_size != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimensions of the 2st input shape should be 1, but got "
                      << input_shape_size;
  }
  grad_input_dim_sizes_ = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename SCALAR_T>
CTask AdaptiveAvgPool3DGradOutFrame(const AdaptiveCalcArgs<SCALAR_T> &args) {
  auto shard_frame = [&args](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      double *grad_input_p_d = args.input_data + d * args.in_size_t * args.in_size_w * args.in_size_h;
      SCALAR_T *grad_output_p_d = args.output_data + d * args.out_size_t * args.out_size_w * args.out_size_h;
      /* calculate average */
      for (int64_t out_t = 0; out_t < args.out_size_t; out_t++) {
        int64_t in_start_t = StartIndex(out_t, args.out_size_t, args.in_size_t);
        int64_t in_end_t = EndIndex(out_t, args.out_size_t, args.in_size_t);
        int64_t span_t = in_end_t - in_start_t;
        for (int64_t out_h = 0; out_h < args.out_size_h; out_h++) {
          int64_t in_start_h = StartIndex(out_h, args.out_size_h, args.in_size_h);
          int64_t in_end_h = EndIndex(out_h, args.out_size_h, args.in_size_h);
          int64_t span_h = in_end_h - in_start_h;
          for (int64_t out_w = 0; out_w < args.out_size_w; out_w++) {
            int64_t in_start_w = StartIndex(out_w, args.out_size_w, args.in_size_w);
            int64_t in_end_w = EndIndex(out_w, args.out_size_w, args.in_size_w);
            int64_t span_w = in_end_w - in_start_w;
            // local pointers
            auto local_grad =
              grad_output_p_d[out_t * args.out_size_h * args.out_size_w + out_h * args.out_size_w + out_w];
            double grad_delta = static_cast<double>(local_grad) / span_t / span_h / span_w;
            for (int64_t in_t = in_start_t; in_t < in_end_t; in_t++) {
              for (int64_t in_h = in_start_h; in_h < in_end_h; in_h++) {
                for (int64_t in_w = in_start_w; in_w < in_end_w; in_w++) {
                  grad_input_p_d[in_t * args.in_size_h * args.in_size_w + in_h * args.in_size_w + in_w] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  };
  return shard_frame;
}

template <typename SCALAR_T>
bool AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  int32_t orig_input_shape_dims = orig_input_shape_dim_sizes_[0];
  auto orig_input_shape_data = static_cast<int32_t *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(orig_input_shape_data);

  AdaptiveCalcArgs<SCALAR_T> args;
  args.size_d = orig_input_shape_data[orig_input_shape_dims + kIdxR4th];
  args.in_size_t = orig_input_shape_data[orig_input_shape_dims + kIdxR3rd];
  args.in_size_h = orig_input_shape_data[orig_input_shape_dims + kIdxR2nd];
  args.in_size_w = orig_input_shape_data[orig_input_shape_dims + kIdxR1st];
  args.out_size_t = grad_output_dim_sizes_[orig_input_shape_dims + kIdxR3rd];
  args.out_size_h = grad_output_dim_sizes_[orig_input_shape_dims + kIdxR2nd];
  args.out_size_w = grad_output_dim_sizes_[orig_input_shape_dims + kIdxR1st];
  auto input_data_ptr_ret = static_cast<SCALAR_T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data_ptr_ret);
  int64_t output_num =
    std::accumulate(grad_input_dim_sizes_.cbegin(), grad_input_dim_sizes_.cend(), 1, std::multiplies<int64_t>{});
  std::unique_ptr<double[]> input_data_ptr = std::make_unique<double[]>(output_num);
  (void)std::fill_n(input_data_ptr.get(), output_num, 0.0);
  auto output_data_ptr = static_cast<SCALAR_T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data_ptr);
  // resize output
  if (orig_input_shape_dims == k4D) {
    args.input_data = input_data_ptr.get();
    args.output_data = output_data_ptr;
    auto shard_frame = AdaptiveAvgPool3DGradOutFrame<SCALAR_T>(args);
    ParallelLaunchAutoSearch(shard_frame, args.size_d, this, &parallel_search_info_);
  } else {
    auto shard_template = [&args, &input_data_ptr, &output_data_ptr, this](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = input_data_ptr.get() + b * args.size_d * args.in_size_t * args.in_size_h * args.in_size_w;
        sub_args.output_data = output_data_ptr + b * args.size_d * args.out_size_t * args.out_size_h * args.out_size_w;
        auto shard_frame = AdaptiveAvgPool3DGradOutFrame<SCALAR_T>(sub_args);
        ParallelLaunchAutoSearch(shard_frame, sub_args.size_d, this, &parallel_search_info_);
      }
    };
    shard_template(0, orig_input_shape_data[0]);
  }
  for (int64_t i = 0; i < output_num; i++) {
    input_data_ptr_ret[i] = static_cast<SCALAR_T>(input_data_ptr[i]);
  }
  return true;
}

std::vector<std::pair<KernelAttr, AdaptiveAvgPool3DGradCPUKernelMod::AdaptiveAvgPool3DGradLaunchFunc>>
  AdaptiveAvgPool3DGradCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveAvgPool3DGradCPUKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> AdaptiveAvgPool3DGradCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, AdaptiveAvgPool3DGradCPUKernelMod::AdaptiveAvgPool3DGradLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveAvgPool3DGrad, AdaptiveAvgPool3DGradCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
