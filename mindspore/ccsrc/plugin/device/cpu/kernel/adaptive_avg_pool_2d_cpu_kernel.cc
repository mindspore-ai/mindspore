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

#include "plugin/device/cpu/kernel/adaptive_avg_pool_2d_cpu_kernel.h"
#include <cmath>
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t k3D = 3;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;
constexpr size_t kIdx1st = 0;
constexpr size_t kIdx2nd = 1;
constexpr int64_t kInputIndex0 = 0;
constexpr int64_t kPyValueNone = -1;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  SCALAR_T *input_data = nullptr;
  SCALAR_T *output_data = nullptr;
  int64_t size_d = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
  int64_t in_stride_d = 0;
  int64_t in_stride_h = 0;
  int64_t in_stride_w = 0;
};

inline int64_t StartIndex(int64_t offset, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(std::floor(static_cast<float>(offset * in_size) / out_size));
}

inline int64_t EndIndex(int64_t offset, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(std::ceil(static_cast<float>((offset + 1) * in_size) / out_size));
}
}  // namespace

void AdaptiveAvgPool2DCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
  node_wpt_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_dim_sizes_ = AnfAlgo::GetInputDeviceShape(kernel_node, kInputIndex0);
  output_size_data_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "output_size");
}

template <typename SCALAR_T>
CTask AdaptiveAvgPool2DOutFrame(const AdaptiveCalcArgs<SCALAR_T> &args) {
  auto shard_frame = [&args](int64_t start, int64_t end) {
    MS_EXCEPTION_IF_NULL(args.input_data);
    MS_EXCEPTION_IF_NULL(args.output_data);
    for (auto d = start; d < end; d++) {
      /* loop over output */
      for (int64_t out_h = 0; out_h < args.out_size_h; out_h++) {
        int64_t in_start_h = StartIndex(out_h, args.out_size_h, args.in_size_h);
        int64_t in_end_h = EndIndex(out_h, args.out_size_h, args.in_size_h);
        int64_t span_h = in_end_h - in_start_h;
        for (int64_t out_w = 0; out_w < args.out_size_w; out_w++) {
          int64_t in_start_w = StartIndex(out_w, args.out_size_w, args.in_size_w);
          int64_t in_end_w = EndIndex(out_w, args.out_size_w, args.in_size_w);
          int64_t span_w = in_end_w - in_start_w;
          // local pointers
          SCALAR_T *in_point =
            args.input_data + d * args.in_stride_d + in_start_h * args.in_stride_h + in_start_w * args.in_stride_w;
          SCALAR_T *out_point =
            args.output_data + d * args.out_size_h * args.out_size_w + out_h * args.out_size_w + out_w;
          /* compute local average */
          double sum = 0;
          for (int64_t in_h = 0; in_h < span_h; in_h++) {
            for (int64_t in_w = 0; in_w < span_w; in_w++) {
              SCALAR_T val = *(in_point + in_h * args.in_stride_h + in_w * args.in_stride_w);
              sum += static_cast<double>(val);
            }
          }
          /* set output to local average */
          *out_point = SCALAR_T(sum / span_h / span_w);
        }
      }
    }
  };
  return shard_frame;
}

bool AdaptiveAvgPool2DCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    (void)LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    (void)LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    (void)LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of input x "
                      << "should be float16 or float32 but got " << TypeIdLabel(dtype_) << ".";
    return false;
  }
  return true;
}

template <typename SCALAR_T>
bool AdaptiveAvgPool2DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto input_size_iter = input_dim_sizes_.rbegin();
  auto output_size_iter = output_size_data_.rbegin();
  for (; output_size_iter != output_size_data_.rend(); output_size_iter++, input_size_iter++) {
    // If output size is none, the input shape should be used.
    if (*output_size_iter == kPyValueNone) {
      *output_size_iter = *input_size_iter;
    }
  }

  size_t input_dims = input_dim_sizes_.size();
  auto input_x = static_cast<SCALAR_T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_x);

  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' node_wpt_, it should not be expired.";
  }

  auto output_y = static_cast<SCALAR_T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_y);

  AdaptiveCalcArgs<SCALAR_T> args;
  constexpr int64_t kIdxSizeD = -3;
  constexpr int64_t kIdxInSizeH = -2;
  constexpr int64_t kIdxInSizeW = -1;
  args.size_d = input_dim_sizes_.end()[kIdxSizeD];
  args.in_size_h = input_dim_sizes_.end()[kIdxInSizeH];
  args.in_size_w = input_dim_sizes_.end()[kIdxInSizeW];
  // strides
  args.in_stride_w = 1;
  args.in_stride_h = args.in_size_w;
  args.in_stride_d = args.in_stride_h * args.in_size_h;

  args.out_size_h = output_size_data_[kIdx1st];
  args.out_size_w = output_size_data_[kIdx2nd];

  // indices will contain i,j locations for each output point
  args.input_data = input_x;
  args.output_data = output_y;
  // resize output
  if (input_dims == k3D) {
    auto shard_frame = AdaptiveAvgPool2DOutFrame<SCALAR_T>(args);
    ParallelLaunchAutoSearch(shard_frame, args.size_d, this, &parallel_search_info_);
  } else {
    auto shard_template = [&args, this](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = args.input_data + b * args.in_stride_d * args.size_d;
        sub_args.output_data = args.output_data + b * args.size_d * args.out_size_h * args.out_size_w;
        auto shard_frame = AdaptiveAvgPool2DOutFrame<SCALAR_T>(sub_args);
        shard_frame(0, sub_args.size_d);
      }
    };
    ParallelLaunchAutoSearch(shard_template, input_dim_sizes_[0], this, &parallel_search_info_);
  }
  return true;
}

std::vector<KernelAttr> AdaptiveAvgPool2DCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveAvgPool2D, AdaptiveAvgPool2DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
