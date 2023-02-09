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
#include "ops/bias_add.h"
#include <map>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBiasAddMinDim = 2;
constexpr size_t kBiasAddMaxDim = 5;
constexpr size_t kBiasAddInputsNum = 2;
constexpr size_t kBiasAddOutputsNum = 1;
}  // namespace

bool BiasAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  return true;
}

int BiasAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  bias_shape_ = Convert2SizeTClipNeg(inputs[kIndex1]->GetShapeVector());
  data_shape_ = input_shape_.size();

  deformable_kernel_operator_ = std::make_shared<ops::BiasAdd>(base_operator->GetPrim());
  data_format_ = deformable_kernel_operator_->get_str_format();
  if ((input_shape_.size() < kBiasAddMinDim || input_shape_.size() > kBiasAddMaxDim) && data_format_ == "NHWC") {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_
      << "', the dimension of 'input_x' tensor must be 2D-5D when data_format is NHWC, but input tensor's dimension is "
      << input_shape_.size();
  }

  if ((input_shape_.size() < kBiasAddMinDim || input_shape_.size() > kBiasAddMaxDim) && data_format_ == "NCHW") {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_
      << "', the dimension of 'input_x' tensor must be 2D-5D when data_format is NCHW, but input tensor's dimension is "
      << input_shape_.size();
  }
  if (input_shape_.size() != kBiasAddMaxDim && (data_format_ == "NCDHW")) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'input_x' tensor must be 5D when data_format is NCDHW, but "
                         "input tensor's dimension is "
                      << input_shape_.size();
  }
  if (bias_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'bias' tensor must be 1D, but got dimension: " << bias_shape_.size();
  }

  if (data_format_ == "NHWC") {
    if (input_shape_[input_shape_.size() - 1] != bias_shape_[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the first dimension length of 'bias' should be equal to "
                           "the last dimension length of 'input_x', but got the first dimension length of 'bias': "
                        << bias_shape_[0]
                        << ", and the last dimension length of 'input_x': " << input_shape_[input_shape_.size() - 1];
    }
  } else if (data_format_ == "NCHW" || data_format_ == "NCDHW") {
    if (input_shape_[1] != bias_shape_[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the first dimension length of 'bias' should be equal to "
                           "the second dimension length of 'input_x', but got the first dimension length of 'bias': "
                        << bias_shape_[0] << ", and the second dimension length of 'input_x': " << input_shape_[1];
    }
  }
  return ret;
}

bool BiasAddCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBiasAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBiasAddOutputsNum, kernel_name_);
  kernel_func_(this, inputs, workspace, outputs);
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
bool BiasAddCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &outputs) {
  const auto *src_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *bias_addr = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
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
      if (data_format_ == "NHWC") {
        ComputeNHWC<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
      } else {
        size_t c_size = input_shape_[kIndex1];
        for (size_t n = 0; n < input_shape_[kIndex0]; ++n) {
          for (size_t c = 0; c < c_size; ++c) {
            size_t offset = n * c_size * hw_size + c * hw_size;
            size_t hw = 0;
#ifdef ENABLE_AVX
            constexpr size_t C8NUM = 8;
            size_t hw8 = hw_size / C8NUM * C8NUM;
            const float *in_ptr = src_addr + offset;
            float *out_ptr = output_addr + offset;
            for (; hw < hw8; hw += C8NUM) {
              __m256 src_r1 = _mm256_loadu_ps(in_ptr);
              __m256 bias_r2 = _mm256_set1_ps(bias_addr[c]);
              __m256 dst_r3 = _mm256_add_ps(src_r1, bias_r2);
              _mm256_storeu_ps(out_ptr, dst_r3);

              in_ptr += C8NUM;
              out_ptr += C8NUM;
            }
#endif
            for (; hw < LongToSize(hw_size); ++hw) {
              output_addr[offset + hw] = src_addr[offset + hw] + bias_addr[c];
            }
          }
        }
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
    if (data_format_ == "NHWC") {
      ComputeNHWC<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
    } else {
      ComputeNCHW<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
    }
  }
  return true;
}

template <typename T>
std::pair<KernelAttr, BiasAddCpuKernelMod::KernelRunFunc> BiasAddCpuKernelMod::MakeKernelFunc(TypeId type_id) const {
  return std::make_pair(KernelAttr().AddInputAttr(type_id).AddInputAttr(type_id).AddOutputAttr(type_id),
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
