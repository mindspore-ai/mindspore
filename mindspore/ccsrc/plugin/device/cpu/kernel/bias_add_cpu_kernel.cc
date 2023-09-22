/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/bias_add_cpu_kernel.h"
#ifdef ENABLE_AVX
#include <immintrin.h>
#endif
#include "ops/ops_func_impl/bias_add.h"
#include <map>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBiasAddMinDim = 2;
constexpr size_t kBiasAddMaxDim = 5;
constexpr size_t kBiasAddInputsNum = 3;
constexpr size_t kBiasAddOutputsNum = 1;
}  // namespace

bool BiasAddCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int BiasAddCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  bias_shape_ = Convert2SizeTClipNeg(inputs[kIndex1]->GetShapeVector());

  data_format_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  if (data_format_ == Format::NCDHW && input_shape_.size() != kShape5dDims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', NCDHW format only supports 5-D input on CPU, but got a "
                             << input_shape_.size() << "-D input.";
  }
  return ret;
}

bool BiasAddCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBiasAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBiasAddOutputsNum, kernel_name_);
  (void)kernel_func_(this, inputs, workspace, outputs);
  return true;
}

template <typename T>
bool BiasAddCpuKernelMod::ComputeNHWC(const T *src_addr, const T *bias_addr, T *output_addr, size_t num_value,
                                      size_t num_bias) {
  if (num_bias == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'bias' tensor shape not be 0, but got : " << num_bias;
  }
  auto task = [this, &output_addr, &src_addr, &bias_addr, num_value, num_bias](size_t start, size_t end) {
    for (size_t i = 0; i < num_value / num_bias; i++) {
      for (size_t j = 0; j < num_bias; j++) {
        size_t it = i * num_bias + j;
        *(output_addr + it) = (*(src_addr + it)) + (*(bias_addr + j));
      }
    }
  };
  ParallelLaunchAutoSearch(task, input_shape_[0], this, &parallel_search_info_);
  return true;
}

template <typename T>
bool BiasAddCpuKernelMod::ComputeNCHW(const T *src_addr, const T *bias_addr, T *output_addr, size_t num_value,
                                      size_t num_bias) {
  if (num_bias == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'bias' tensor shape not be 0, but got : " << num_bias;
  }
  auto len = num_value / input_shape_[0] / num_bias;
  auto task = [this, &output_addr, &src_addr, &bias_addr, num_value, num_bias, len](size_t start, size_t end) {
    for (size_t i = 0; i < input_shape_[0]; i++) {
      for (size_t j = 0; j < num_bias; j++) {
        for (size_t k = 0; k < len; k++) {
          size_t it = i * num_bias * len + j * len + k;
          *(output_addr + it) = (*(src_addr + it)) + (*(bias_addr + j));
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, input_shape_[0], this, &parallel_search_info_);
  return true;
}

template <typename T>
bool BiasAddCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs) {
  const auto *src_addr = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  const auto *bias_addr = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());
  auto *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  size_t num_value = 1;
  size_t num_bias = bias_shape_[0];
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    num_value *= input_shape_[i];
  }
  if constexpr (std::is_same_v<T, float>) {
    if (input_shape_.size() > kBiasAddMinDim) {
      size_t hw_size = 1;
      for (size_t i = 2; i < input_shape_.size(); ++i) {
        hw_size *= input_shape_[i];
      }
      if (data_format_ == Format::NHWC) {
        ComputeNHWC<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
      } else {
        size_t c_size = input_shape_[kIndex1];
        auto task = [&](size_t start, size_t end) {
          for (size_t n = start; n < end; ++n) {
            for (size_t c = 0; c < c_size; ++c) {
              size_t offset = LongToSize(n * c_size * hw_size + c * hw_size);
              size_t hw = 0;
              for (; hw < LongToSize(hw_size); ++hw) {
                output_addr[offset + hw] = src_addr[offset + hw] + bias_addr[c];
              }
            }
          }
        };
        ParallelLaunchAutoSearch(task, LongToSize(input_shape_[0]), this, &parallel_search_info_);
      }
    } else {
      auto task = [&](size_t start, size_t end) {
        for (size_t n = start; n < end; ++n) {
          size_t n_offset = LongToSize(input_shape_[kIndex1] * n);
          const T *inner_src = src_addr + n_offset;
          T *inner_dst = output_addr + n_offset;
          for (size_t index = 0; index < input_shape_[kIndex1]; ++index) {
            inner_dst[index] = inner_src[index] + bias_addr[index];
          }
        }
      };
      ParallelLaunchAutoSearch(task, LongToSize(input_shape_[kIndex0]), this, &parallel_search_info_);
    }
  } else {
    if (data_format_ == Format::NHWC) {
      ComputeNHWC<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
    } else {
      ComputeNCHW<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
    }
  }
  return true;
}

template <typename T>
std::pair<KernelAttr, BiasAddCpuKernelMod::KernelRunFunc> BiasAddCpuKernelMod::MakeKernelFunc(TypeId type_id) const {
  return std::make_pair(KernelAttr()
                          .AddInputAttr(type_id)
                          .AddInputAttr(type_id)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(type_id),
                        &BiasAddCpuKernelMod::LaunchKernel<T>);
}

const std::vector<std::pair<KernelAttr, BiasAddCpuKernelMod::KernelRunFunc>> &BiasAddCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BiasAddCpuKernelMod::KernelRunFunc>> func_list = {
    MakeKernelFunc<float>(kNumberTypeFloat32),
    MakeKernelFunc<double>(kNumberTypeFloat64),
    MakeKernelFunc<int8_t>(kNumberTypeInt8),
    MakeKernelFunc<int16_t>(kNumberTypeInt16),
    MakeKernelFunc<int32_t>(kNumberTypeInt32),
    MakeKernelFunc<int64_t>(kNumberTypeInt64),
    MakeKernelFunc<uint8_t>(kNumberTypeUInt8),
    MakeKernelFunc<uint16_t>(kNumberTypeUInt16),
    MakeKernelFunc<uint32_t>(kNumberTypeUInt32),
    MakeKernelFunc<uint64_t>(kNumberTypeUInt64),
    MakeKernelFunc<std::complex<float>>(kNumberTypeComplex64),
    MakeKernelFunc<std::complex<double>>(kNumberTypeComplex128),
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BiasAdd, BiasAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
