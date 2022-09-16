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

#include "plugin/device/cpu/kernel/adaptive_avg_pool_2d_grad_v1_cpu_kernel.h"
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t k4D = 4;
constexpr size_t k3D = 3;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;
constexpr int64_t kIdxR3rd = -3;
constexpr int64_t kIdxR2nd = -2;
constexpr int64_t kIdxR1st = -1;
constexpr int64_t kInputIndex0 = 0;
constexpr int64_t kOutputIndex0 = 0;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  double *input_data = nullptr;
  SCALAR_T *output_data = nullptr;
  int64_t size_d = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
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

void AdaptiveAvgPool2DGradV1CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
  grad_output_dim_sizes = AnfAlgo::GetInputDeviceShape(kernel_node, kInputIndex0);
  orig_input_shape_dim_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "orig_input_shape");
  grad_input_dim_sizes = AnfAlgo::GetOutputDeviceShape(kernel_node, kOutputIndex0);
}

template <typename SCALAR_T>
CTask AdaptiveAvgPool2DGradV1OutFrame(const AdaptiveCalcArgs<SCALAR_T> &args) {
  auto shard_frame = [&args](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      double *grad_input_p_d = args.input_data + d * args.in_size_w * args.in_size_h;
      SCALAR_T *grad_output_p_d = args.output_data + d * args.out_size_w * args.out_size_h;
      /* calculate average */
      for (int64_t out_h = 0; out_h < args.out_size_h; out_h++) {
        int64_t in_start_h = StartIndex(out_h, args.out_size_h, args.in_size_h);
        int64_t in_end_h = EndIndex(out_h, args.out_size_h, args.in_size_h);
        int64_t span_h = in_end_h - in_start_h;
        for (int64_t out_w = 0; out_w < args.out_size_w; out_w++) {
          int64_t in_start_w = StartIndex(out_w, args.out_size_w, args.in_size_w);
          int64_t in_end_w = EndIndex(out_w, args.out_size_w, args.in_size_w);
          int64_t span_w = in_end_w - in_start_w;
          // local pointers
          auto local_grad = grad_output_p_d[out_h * args.out_size_w + out_w];
          double grad_delta = static_cast<double>(local_grad) / span_h / span_w;
          for (int64_t in_h = in_start_h; in_h < in_end_h; in_h++) {
            for (int64_t in_w = in_start_w; in_w < in_end_w; in_w++) {
              grad_input_p_d[in_h * args.in_size_w + in_w] += grad_delta;
            }
          }
        }
      }
    }
  };
  return shard_frame;
}

bool AdaptiveAvgPool2DGradV1CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of input x "
                      << "should be float16 or float32 but got " << TypeIdLabel(dtype_) << ".";
    return false;
  }
  return true;
}

template <typename SCALAR_T>
bool AdaptiveAvgPool2DGradV1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  int32_t orig_input_shape_dims = orig_input_shape_dim_sizes.size();
  if (orig_input_shape_dims != k3D && orig_input_shape_dims != k4D) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', Non-empty [3D] or [4D] tensor expected for orig_input, "
                         "but got dimension size "
                      << orig_input_shape_dims;
  }
  for (int32_t i = 0; i < orig_input_shape_dims; i++) {
    if (orig_input_shape_dim_sizes[i] <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', expected orig_input to have "
                           "non-empty spatial dimensions, but got "
                        << orig_input_shape_dim_sizes[i];
    }
  }
  size_t grad_output_dims = grad_output_dim_sizes.size();
  if (grad_output_dims != k3D && grad_output_dims != k4D) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', Non-empty [3D] or [4D] tensor expected for input_grad, "
                         "but got dimension size"
                      << grad_output_dims;
  }
  AdaptiveCalcArgs<SCALAR_T> args;
  args.size_d = orig_input_shape_dim_sizes[orig_input_shape_dims + kIdxR3rd];
  args.in_size_h = orig_input_shape_dim_sizes[orig_input_shape_dims + kIdxR2nd];
  args.in_size_w = orig_input_shape_dim_sizes[orig_input_shape_dims + kIdxR1st];
  args.out_size_h = grad_output_dim_sizes.end()[kIdxR2nd];
  args.out_size_w = grad_output_dim_sizes.end()[kIdxR1st];
  auto input_data_ptr_ret = reinterpret_cast<SCALAR_T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data_ptr_ret);
  int64_t output_num =
    std::accumulate(grad_input_dim_sizes.cbegin(), grad_input_dim_sizes.cend(), 1, std::multiplies<int64_t>{});
  std::unique_ptr<double[]> input_data_ptr = std::make_unique<double[]>(output_num);
  std::fill_n(input_data_ptr.get(), output_num, 0.0);
  auto output_data_ptr = reinterpret_cast<SCALAR_T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data_ptr);
  // resize output
  if (orig_input_shape_dims == k3D) {
    args.input_data = input_data_ptr.get();
    args.output_data = output_data_ptr;
    auto shard_frame = AdaptiveAvgPool2DGradV1OutFrame<SCALAR_T>(args);
    ParallelLaunchAutoSearch(shard_frame, args.size_d, this, &parallel_search_info_);
  } else {
    auto shard_template = [&args, &input_data_ptr, &output_data_ptr, this](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = input_data_ptr.get() + b * args.size_d * args.in_size_h * args.in_size_w;
        sub_args.output_data = output_data_ptr + b * args.size_d * args.out_size_h * args.out_size_w;
        auto shard_frame = AdaptiveAvgPool2DGradV1OutFrame<SCALAR_T>(sub_args);
        ParallelLaunchAutoSearch(shard_frame, sub_args.size_d, this, &parallel_search_info_);
      }
    };
    ParallelLaunchAutoSearch(shard_template, orig_input_shape_dim_sizes[0], this, &parallel_search_info_);
  }
  for (int64_t i = 0; i < output_num; i++) {
    input_data_ptr_ret[i] = static_cast<SCALAR_T>(input_data_ptr[i]);
  }
  return true;
}

std::vector<KernelAttr> AdaptiveAvgPool2DGradV1CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveAvgPool2DGradV1, AdaptiveAvgPool2DGradV1CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
