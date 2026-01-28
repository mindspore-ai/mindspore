/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/select_gpu_kernel.h"

#include <utility>
#include <vector>

#include "mindspore/ops/infer/ops_func_impl/select.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool SelectGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<kernel::KernelTensor *> &workspace,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_cond = reinterpret_cast<bool *>(inputs[0]->device_ptr());
  auto *input_x = reinterpret_cast<T *>(inputs[1]->device_ptr());
  auto *input_y = reinterpret_cast<T *>(inputs[2]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  auto status =
    is_need_broadcast_
      ? CalSelectBroadcast(input_cond, input_x, input_y, output, dim_size_, output_size_, strides_, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream_))
      : CalSelect(input_cond, input_x, input_y, output, output_size_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

using selectPair = std::pair<KernelAttr, SelectGpuKernelMod::KernelRunFunc>;
const std::vector<selectPair> &SelectGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SelectGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SelectGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SelectGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SelectGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &SelectGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &SelectGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SelectGpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SelectGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &SelectGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &SelectGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &SelectGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SelectGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &SelectGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SelectGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SelectGpuKernelMod::LaunchKernel<utils::Complex<double>>},
  };
  return func_list;
}

bool SelectGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int SelectGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }

  auto output_shape = outputs[0]->GetShapeVector();
  auto output_shape_size_t = Convert2SizeTClipNeg(output_shape);
  is_null_input_ = CHECK_SHAPE_NULL(output_shape_size_t, kernel_name_, "output");
  if (is_null_input_) {
    return true;
  }
  size_t input_num = inputs.size();
  if (input_num != kSelectInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
    return KRET_RESIZE_FAILED;
  }
  size_t output_num = outputs.size();
  if (output_num != kSelectOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    return KRET_RESIZE_FAILED;
  }
  output_size_ = kSelectOutputSizeInit;
  for (size_t x : output_shape_size_t) {
    output_size_ = output_size_ * x;
  }
  auto cond_shape = inputs[kIndex0]->GetShapeVector();
  auto x_shape = inputs[kIndex1]->GetShapeVector();
  auto y_shape = inputs[kIndex2]->GetShapeVector();
  is_need_broadcast_ = (cond_shape != output_shape) || (x_shape != output_shape) || (y_shape != output_shape);
  if (is_need_broadcast_) {
    constexpr size_t kMaxBroadcastDim = 8;
    dim_size_ = output_shape.size();
    if (dim_size_ > kMaxBroadcastDim) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the broadcast dim size must be <= " << kMaxBroadcastDim
                        << ", but got " << dim_size_;
    }
    auto AlignShape = [](const std::vector<int64_t> &shape, size_t dim_size) {
      std::vector<int64_t> aligned(dim_size, 1);
      if (shape.empty()) {
        return aligned;
      }
      size_t offset = dim_size - shape.size();
      for (size_t i = 0; i < shape.size(); ++i) {
        aligned[offset + i] = shape[i];
      }
      return aligned;
    };
    auto cond_shape_aligned = AlignShape(cond_shape, dim_size_);
    auto x_shape_aligned = AlignShape(x_shape, dim_size_);
    auto y_shape_aligned = AlignShape(y_shape, dim_size_);
    auto out_shape_aligned = AlignShape(output_shape, dim_size_);
    strides_ =
      TrinaryBroadcastCalStride(dim_size_, cond_shape_aligned, x_shape_aligned, y_shape_aligned, out_shape_aligned);
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Select, SelectGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
