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
constexpr size_t kInputNumDims5 = 5;
constexpr size_t kInputShapeDims4 = 4;

bool AdaptiveMaxPool3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  is_need_retrieve_output_shape_ = true;
  return true;
}

int AdaptiveMaxPool3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_num_dims_ = input_shape_.size();
  outputs_ = outputs;

  if (!(input_num_dims_ == kInputNumDims5 || input_num_dims_ == kInputShapeDims4)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input data dimensions should be equal to 4 or 5, but got "
                  << input_num_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  auto output_size_shape = inputs[kIndex1]->GetShapeVector();
  if (output_size_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size dimensions should be equal to 1, but got "
                  << output_size_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  const size_t kOutputSizeElemNum = 3;
  if (output_size_shape[0] != kOutputSizeElemNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size elem number should be equal to 3, but got "
                  << output_size_shape[0] << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

void AdaptiveMaxPool3DCpuKernelMod::SyncData() {
  outputs_[kIndex0]->SetShapeVector(output_shape_);
  outputs_[kIndex1]->SetShapeVector(output_shape_);
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
  output_shape_ = {input_shape_[0]};
  if (input_num_dims_ == kInputNumDims5) {
    output_shape_.push_back(input_shape_[1]);
  }
  auto output_size_ptr = reinterpret_cast<int32_t *>(inputs[1]->addr);
  const size_t kOutputSizeDims = 3;
  for (size_t i = 0; i < kOutputSizeDims; ++i) {
    const int32_t elem = output_size_ptr[i];
    if (elem <= 0) {
      MS_EXCEPTION(ValueError) << "For AdaptiveMaxPool3D, elements of output_size must be greater than 0, but got "
                               << elem << ".";
    }
    output_shape_.push_back(static_cast<size_t>(elem));
  }

  switch (dtype_) {
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
  if (input_shape_.size() == kInputShapeDims4) {
    input_shape_.insert(input_shape_.begin(), 1);
  }
  size_B_ = input_shape_[dimB];
  size_D_ = input_shape_[dimD];
  input_size_T_ = input_shape_[dimT];
  input_size_H_ = input_shape_[dimH];
  input_size_W_ = input_shape_[dimW];
  input_stride_B_ = ComputeStride(input_shape_, dimB);
  input_stride_D_ = ComputeStride(input_shape_, dimD);
  input_stride_T_ = ComputeStride(input_shape_, dimT);
  input_stride_H_ = ComputeStride(input_shape_, dimH);
  input_stride_W_ = ComputeStride(input_shape_, dimW);
  const ptrdiff_t kIndexT = 3;
  const ptrdiff_t kIndexH = 2;
  output_size_T_ = output_shape_.cend()[-kIndexT];
  output_size_H_ = output_shape_.cend()[-kIndexH];
  output_size_W_ = output_shape_.cend()[-1];

  auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
    ComputeKernel(input_data, output_data, indices_data, start, end);
  };

  // The AdaptiveMaxPool3D will be reinit in graph mode, so the ParallelLaunchAutoSearch dose not work, use
  // ParallelLaunch instead.
  const float block_size = 1.0;
  ParallelLaunch(shard_adaptive_max_pool_3d, output_size_T_, block_size);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool3D, AdaptiveMaxPool3DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
