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
#include "plugin/device/cpu/kernel/adaptive_max_pool_3d_cpu_kernel.h"

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
#define ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(TYPENUM, DTYPE) \
  case (TYPENUM): {                                       \
    AdaptiveMaxPool3DCompute<DTYPE>(inputs, outputs);     \
    break;                                                \
  }
}  // namespace

namespace mindspore {
namespace kernel {
void AdaptiveMaxPool3DCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  const size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  const size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, input number is " << input_num
                      << ", but AdaptiveMaxPool3DCPUKernel needs " << kInputNum << " input.";
  }
  if (output_num != kOutputNum) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output number is " << output_num
                      << ", but AdaptiveMaxPool3DCPUKernel needs " << kOutputNum << " output.";
  }
  input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_num_dims_ = input_shape.size();
  const size_t kNumDims4 = 4;
  const size_t kNumDims5 = 5;
  if (!(input_num_dims_ == kNumDims4 || input_num_dims_ == kNumDims5)) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, input data dimensions should be equal to 4 or 5, but got "
                      << input_num_dims_ << ".";
  }
  output_size_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (output_size_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output size dimensions should be equal to 1, but got "
                      << output_size_shape.size() << ".";
  }
  const size_t kOutputSizeElemNum = 3;
  if (output_size_shape[0] != kOutputSizeElemNum) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output size elem number should be equal to 3, but got "
                      << output_size_shape[0] << ".";
  }
  dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (AnfAlgo::GetInputDeviceDataType(kernel_node, 1) != kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output size dtype must be int32.";
  }
}

std::vector<KernelAttr> AdaptiveMaxPool3DCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeUInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeUInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}

bool AdaptiveMaxPool3DCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  // Set Shape
  const size_t kInputNumDims5 = 5;
  output_shape = {input_shape[0]};
  if (input_num_dims_ == kInputNumDims5) {
    output_shape.push_back(input_shape[1]);
  }
  auto output_size_ptr = reinterpret_cast<int32_t *>(inputs[1]->addr);
  const size_t kOutputSizeDims = 3;
  for (size_t i = 0; i < kOutputSizeDims; ++i) {
    const int32_t elem = output_size_ptr[i];
    if (elem <= 0) {
      MS_EXCEPTION(ValueError) << "For AdaptiveMaxPool3D, elements of output_size must be greater than 0, but got "
                               << elem << ".";
    }
    output_shape.push_back(static_cast<size_t>(elem));
  }
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype, kNumberTypeInt32}, {output_shape, output_shape},
                                              node_wpt_.lock().get());

  switch (dtype) {
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeUInt32, uint32_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeUInt64, uint64_t)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeFloat16, float16)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeFloat32, float)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(kNumberTypeFloat64, double)
    default:
      MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, kernel data type not support.";
      return false;
  }
  return true;
}

int64_t AdaptiveMaxPool3DCpuKernelMod::ComputeStride(const std::vector<int64_t> &shape, size_t index) {
  if (index >= shape.size()) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, input index must be less than shape dims.";
  }
  int64_t result = 1;
  for (size_t i = index + 1; i < shape.size(); ++i) {
    result *= shape[i];
  }
  return result;
}
template <typename T>
void AdaptiveMaxPool3DCpuKernelMod::ComputeKernel(T *input_data, T *output_data, int32_t *indices_data, int64_t start_T,
                                                  int64_t end_T) {
  auto start_index = [=](int64_t dim, int64_t output_range, int64_t input_range) {
    if (output_range == 0) {
      MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output range should not be zero.";
    }
    return static_cast<int64_t>(std::floor(static_cast<double>(dim * input_range) / output_range));
  };
  auto end_index = [=](int64_t dim, int64_t output_range, int64_t input_range) {
    if (output_range == 0) {
      MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3D, output range should not be zero.";
    }
    return static_cast<int64_t>(std::ceil(static_cast<double>((dim + 1) * input_range) / output_range));
  };
  for (int64_t b = 0; b < size_B_; ++b) {
    auto input_ptr = input_data + b * input_stride_B_;
    auto output_ptr = output_data + b * size_D_ * output_size_T_ * output_size_H_ * output_size_W_;
    auto indice_ptr = indices_data + b * size_D_ * output_size_T_ * output_size_H_ * output_size_W_;
    for (int64_t d = 0; d < size_D_; ++d) {
      int64_t ot, oh, ow;
      for (ot = start_T; ot < end_T; ++ot) {
        int64_t input_start_T = start_index(ot, output_size_T_, input_size_T_);
        int64_t input_end_T = end_index(ot, output_size_T_, input_size_T_);
        int64_t kT = input_end_T - input_start_T;
        for (oh = 0; oh < output_size_H_; ++oh) {
          int64_t input_start_H = start_index(oh, output_size_H_, input_size_H_);
          int64_t input_end_H = end_index(oh, output_size_H_, input_size_H_);
          int64_t kH = input_end_H - input_start_H;
          for (ow = 0; ow < output_size_W_; ++ow) {
            int64_t input_start_W = start_index(ow, output_size_W_, input_size_W_);
            int64_t input_end_W = end_index(ow, output_size_W_, input_size_W_);
            int64_t kW = input_end_W - input_start_W;
            auto ip = input_ptr + d * input_stride_D_ + input_start_T * input_stride_T_ +
                      input_start_H * input_stride_H_ + input_start_W * input_stride_W_;
            auto op = output_ptr + d * output_size_T_ * output_size_H_ * output_size_W_ +
                      ot * output_size_H_ * output_size_W_ + oh * output_size_W_ + ow;
            auto indp = indice_ptr + d * output_size_T_ * output_size_H_ * output_size_W_ +
                        ot * output_size_H_ * output_size_W_ + oh * output_size_W_ + ow;
            int64_t it = 0, ih = 0, iw = 0;
            int64_t maxindex = (it + input_start_T) * input_size_H_ * input_size_W_ +
                               (ih + input_start_H) * input_size_W_ + (iw + input_start_W);
            T maxval = *(ip + it * input_stride_T_ + ih * input_stride_H_ + iw * input_stride_W_);
            for (it = 0; it < kT; ++it) {
              for (ih = 0; ih < kH; ++ih) {
                for (iw = 0; iw < kW; ++iw) {
                  T val = *(ip + it * input_stride_T_ + ih * input_stride_H_ + iw * input_stride_W_);
                  if (val > maxval) {
                    maxval = val;
                    maxindex = (it + input_start_T) * input_size_H_ * input_size_W_ +
                               (ih + input_start_H) * input_size_W_ + (iw + input_start_W);
                  }
                }
              }
            }
            *op = maxval;
            *indp = static_cast<int32_t>(maxindex);
          }
        }
      }
    }
  }
}

template <typename T>
void AdaptiveMaxPool3DCpuKernelMod::AdaptiveMaxPool3DCompute(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &outputs) {
  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_data = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices_data = reinterpret_cast<int32_t *>(outputs[1]->addr);
  const size_t kInputShapeDims4 = 4;
  if (input_shape.size() == kInputShapeDims4) {
    input_shape.insert(input_shape.begin(), 1);
    output_shape.insert(output_shape.begin(), 1);
  }
  size_B_ = input_shape[dimB];
  size_D_ = input_shape[dimD];
  input_size_T_ = input_shape[dimT];
  input_size_H_ = input_shape[dimH];
  input_size_W_ = input_shape[dimW];
  input_stride_B_ = ComputeStride(input_shape, dimB);
  input_stride_D_ = ComputeStride(input_shape, dimD);
  input_stride_T_ = ComputeStride(input_shape, dimT);
  input_stride_H_ = ComputeStride(input_shape, dimH);
  input_stride_W_ = ComputeStride(input_shape, dimW);
  const ptrdiff_t kIndexT = 3;
  const ptrdiff_t kIndexH = 2;
  output_size_T_ = output_shape.cend()[-kIndexT];
  output_size_H_ = output_shape.cend()[-kIndexH];
  output_size_W_ = output_shape.cend()[-1];

  auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
    ComputeKernel(input_data, output_data, indices_data, start, end);
  };
  CPUKernelUtils::ParallelFor(shard_adaptive_max_pool_3d, output_size_T_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool3D, AdaptiveMaxPool3DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
