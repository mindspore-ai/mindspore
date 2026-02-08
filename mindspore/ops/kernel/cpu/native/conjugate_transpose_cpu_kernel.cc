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

#include <complex>
#include <cstdint>
#include <vector>
#include "mindapi/base/type_id.h"
#include "nnacl/errorcode.h"
#include "include/utils/thread_pool.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"
#include "kernel/cpu/native/conjugate_transpose_cpu_kernel.h"

#include "utils/convert_utils_base.h"

namespace mindspore {
namespace kernel {
namespace conjugate_transpose_cpu {
namespace {
constexpr size_t kConjugateTransposeInputsNum = 2;
constexpr size_t kConjugateTransposeOutputsNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kMaxTransposeSerialSize = 50331648;
}  // namespace

bool ConjugateTransposeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  dtype_ = inputs.at(kIndex0)->dtype_id();
  perm_type_ = inputs.at(kIndex1)->dtype_id();
  launch_map_[kNumberTypeBool] = &ConjugateTransposeCpuKernelMod::LaunchKernel<bool>;
  launch_map_[kNumberTypeInt8] = &ConjugateTransposeCpuKernelMod::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &ConjugateTransposeCpuKernelMod::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &ConjugateTransposeCpuKernelMod::LaunchKernel<int32_t>;
  launch_map_[kNumberTypeInt64] = &ConjugateTransposeCpuKernelMod::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &ConjugateTransposeCpuKernelMod::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &ConjugateTransposeCpuKernelMod::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &ConjugateTransposeCpuKernelMod::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &ConjugateTransposeCpuKernelMod::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat16] = &ConjugateTransposeCpuKernelMod::LaunchKernel<float16>;
  launch_map_[kNumberTypeFloat32] = &ConjugateTransposeCpuKernelMod::LaunchKernel<float>;
  launch_map_[kNumberTypeFloat64] = &ConjugateTransposeCpuKernelMod::LaunchKernel<double>;
  launch_map_[kNumberTypeComplex64] = &ConjugateTransposeCpuKernelMod::LaunchComplexKernel<complex64>;
  launch_map_[kNumberTypeComplex128] = &ConjugateTransposeCpuKernelMod::LaunchComplexKernel<complex128>;
  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "For ConjugateTranspose: unsupported input data type: " << dtype_;
  }
  return true;
}

int ConjugateTransposeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs.at(kIndex0)->GetDeviceShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetDeviceShapeVector();

  return KRET_OK;
}

bool ConjugateTransposeCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConjugateTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConjugateTransposeOutputsNum, kernel_name_);
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void ConjugateTransposeCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  const auto *input_addr = static_cast<T *>(inputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_addr);

  if (perm_type_ == kNumberTypeInt32) {
    auto perm_addr = static_cast<int32_t *>(inputs[1]->device_ptr());
    MS_EXCEPTION_IF_NULL(perm_addr);
    auto perm_size = SizeToInt(inputs[1]->size() / sizeof(int32_t));
    for (int i = 0; i < perm_size; ++i) {
      auto p = perm_addr[i];
      p = (p >= 0) ? p : (perm_size + p);
      if (p < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm_size << ", "
                          << (perm_size - 1) << "], but got " << p << " .";
      }
      (void)axes_.emplace_back(p);
    }
  } else if (perm_type_ == kNumberTypeInt64) {
    auto perm_addr = static_cast<int64_t *>(inputs[1]->device_ptr());
    MS_EXCEPTION_IF_NULL(perm_addr);
    auto perm_size = SizeToInt(inputs[1]->size() / sizeof(int64_t));
    for (int i = 0; i < perm_size; ++i) {
      auto p = perm_addr[i];
      p = (p >= 0) ? p : (perm_size + p);
      if (p < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm_size << ", "
                          << (perm_size - 1) << "], but got " << p << " .";
      }
      (void)axes_.emplace_back(p);
    }
  }

  if (axes_.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(EXCEPTION) << "ConjugateTranspose support max dimension is " << MAX_TRANSPOSE_DIM_SIZE << "D, but got "
                      << axes_.size() << "D.";
  }
  for (size_t i = 0; i < axes_.size(); ++i) {
    transpose_param_.perm_[i] = SizeToInt(axes_[i]);
  }
  size_t num_axes = input_shape_.size();
  if (num_axes == 0) {
    MS_EXCEPTION(ValueError) << "ConjugateTranspose doesn't support input shape's size as 0.";
  }
  transpose_param_.perm_size_ = axes_.size();
  transpose_param_.num_axes_ = SizeToInt(num_axes);
  transpose_param_.strides_[num_axes - 1] = 1;
  transpose_param_.out_strides_[num_axes - 1] = 1;
  for (size_t i = num_axes - 1; i >= 1; i--) {
    transpose_param_.strides_[i - 1] = SizeToInt(input_shape_[i]) * transpose_param_.strides_[i];
    transpose_param_.out_strides_[i - 1] = SizeToInt(output_shape_[i]) * transpose_param_.out_strides_[i];
  }

  auto *output_addr = static_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_addr);
  transpose_param_.data_num_ = SizeToInt(inputs[0]->size() / sizeof(T));
  std::vector<int> output_shape(SizeToInt(output_shape_.size()));
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_shape[i] = SizeToInt(output_shape_[i]);
  }
  size_t data_count = (inputs[0]->size()) / sizeof(T);
  if (axes_.size() > kIndex7 || data_count >= kMaxTransposeSerialSize) {
    ParallelRun(input_addr, output_addr, output_shape.data(), data_count, &transpose_param_);
    return;
  }
  bool res{static_cast<bool>(NNACL_OK)};
  res = DoTranspose(input_addr, output_addr, output_shape.data(), &transpose_param_);
  if (res != static_cast<bool>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "ConjugateTranspose run failed.";
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::ConjComplexFunc(T *input, T *output, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    output[i] = static_cast<T>(std::conj(input[i]));
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::LaunchComplexKernel(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  auto *input_addr = static_cast<T *>(inputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_addr);

  if (perm_type_ == kNumberTypeInt32) {
    auto perm_addr = static_cast<int32_t *>(inputs[1]->device_ptr());
    MS_EXCEPTION_IF_NULL(perm_addr);
    auto perm_size = SizeToInt(inputs[1]->size() / sizeof(int32_t));
    for (int i = 0; i < perm_size; ++i) {
      auto p = perm_addr[i];
      p = (p >= 0) ? p : (perm_size + p);
      if (p < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm_size << ", "
                          << (perm_size - 1) << "], but got " << p << " .";
      }
      (void)axes_.emplace_back(p);
    }
  } else if (perm_type_ == kNumberTypeInt64) {
    auto perm_addr = static_cast<int64_t *>(inputs[1]->device_ptr());
    MS_EXCEPTION_IF_NULL(perm_addr);
    auto perm_size = SizeToInt(inputs[1]->size() / sizeof(int64_t));
    for (int i = 0; i < perm_size; ++i) {
      auto p = perm_addr[i];
      p = (p >= 0) ? p : (perm_size + p);
      if (p < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm_size << ", "
                          << (perm_size - 1) << "], but got " << p << " .";
      }
      (void)axes_.emplace_back(p);
    }
  }

  if (axes_.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(EXCEPTION) << "ConjugateTranspose support max dimension is " << MAX_TRANSPOSE_DIM_SIZE << "D, but got "
                      << axes_.size() << "D.";
  }
  for (size_t i = 0; i < axes_.size(); ++i) {
    transpose_param_.perm_[i] = SizeToInt(axes_[i]);
  }
  size_t num_axes = input_shape_.size();
  if (num_axes == 0) {
    MS_EXCEPTION(ValueError) << "ConjugateTranspose doesn't support input shape's size as 0.";
  }
  transpose_param_.perm_size_ = axes_.size();
  transpose_param_.num_axes_ = SizeToInt(num_axes);
  transpose_param_.strides_[num_axes - 1] = 1;
  transpose_param_.out_strides_[num_axes - 1] = 1;
  for (size_t i = num_axes - 1; i >= 1; i--) {
    transpose_param_.strides_[i - 1] = SizeToInt(input_shape_[i]) * transpose_param_.strides_[i];
    transpose_param_.out_strides_[i - 1] = SizeToInt(output_shape_[i]) * transpose_param_.out_strides_[i];
  }

  auto *output_addr = static_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_addr);
  transpose_param_.data_num_ = SizeToInt(inputs[0]->size() / sizeof(T));
  auto task = std::bind(ConjComplexFunc<T>, input_addr, input_addr, 0, transpose_param_.data_num_);
  ParallelLaunchAutoSearch(task, transpose_param_.data_num_, this, &parallel_search_info_);

  std::vector<int> output_shape(SizeToInt(output_shape_.size()));
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_shape[i] = SizeToInt(output_shape_[i]);
  }
  size_t data_count = (inputs[0]->size()) / sizeof(T);
  if (axes_.size() > kIndex7 || data_count >= kMaxTransposeSerialSize) {
    ParallelRun(input_addr, output_addr, output_shape.data(), data_count, &transpose_param_);
    return;
  }
  bool res{static_cast<bool>(NNACL_OK)};
  res = DoTranspose(input_addr, output_addr, output_shape.data(), &transpose_param_);
  if (res != static_cast<bool>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "ConjugateTranspose run failed.";
  }
}

template <typename T>
int ConjugateTransposeCpuKernelMod::DoTranspose(const T *in_data, T *out_data, const int *output_shape,
                                                const TransposeParameter *transpose_param) {
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  NNACL_CHECK_NULL_RETURN_ERR(output_shape);
  NNACL_CHECK_NULL_RETURN_ERR(transpose_param);
  const int *perm = transpose_param->perm_;
  const int *strides = transpose_param->strides_;
  const int *out_strides = transpose_param->out_strides_;
  data_size = static_cast<int32_t>(transpose_param->data_num_) * static_cast<int32_t>(sizeof(T));
  int num_axes = transpose_param->num_axes_;
  bool needTranspose = false;
  for (size_t i = 1; i < static_cast<uint32_t>(num_axes); ++i) {
    if (perm[i] - perm[i - 1] != 1) {
      needTranspose = true;
      break;
    }
  }
  if (!needTranspose) {
    auto ret = memcpy_s(out_data, data_size, in_data, data_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s failed, ret=" << ret;
    }
    return NNACL_OK;
  }
  for (size_t i = 0; i < static_cast<uint32_t>(num_axes); ++i) {
    if (perm[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  if (num_axes == kIndex2) {
    TransposeDim2(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == kIndex3) {
    TransposeDim3(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == kIndex4) {
    TransposeDim4(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == kIndex5) {
    TransposeDim5(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == kIndex6) {
    TransposeDim6(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == kIndex7) {
    TransposeDim7(in_data, out_data, strides, out_strides, perm, output_shape);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim2(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * output1;
    size_t stride0_i = i * 1 * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      out_data[out_stride0_i + j] = in_data[stride0_i + j * stride1];
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim3(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < static_cast<uint32_t>(output2); ++k) {
        out_data[out_stride0_i + out_stride1_j + k] = in_data[stride0_i + stride1_j + k * stride2];
      }
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim4(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int stride3 = strides[perm[kIndex3]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int out_stride2 = out_strides[kIndex2];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  const int output3 = output_shape[kIndex3];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < static_cast<uint32_t>(output2); ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < static_cast<uint32_t>(output3); ++m) {
          int num = SizeToLong(stride0_i + stride1_j + stride2_k + m * stride3);
          if (num >= data_size) {
            MS_LOG(EXCEPTION) << "For 'ConjugateTranspose', dimension of input data exceed data size." << num;
          }
          out_data[out_stride0_i + out_stride1_j + out_stride2_k + m] = in_data[num];
        }
      }
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim5(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int stride3 = strides[perm[kIndex3]];
  const int stride4 = strides[perm[kIndex4]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int out_stride2 = out_strides[kIndex2];
  const int out_stride3 = out_strides[kIndex3];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  const int output3 = output_shape[kIndex3];
  const int output4 = output_shape[kIndex4];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < static_cast<uint32_t>(output2); ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < static_cast<uint32_t>(output3); ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < static_cast<uint32_t>(output4); ++n) {
            out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + n] =
              in_data[stride0_i + stride1_j + stride2_k + stride3_m + n * stride4];
          }
        }
      }
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim6(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int stride3 = strides[perm[kIndex3]];
  const int stride4 = strides[perm[kIndex4]];
  const int stride5 = strides[perm[kIndex5]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int out_stride2 = out_strides[kIndex2];
  const int out_stride3 = out_strides[kIndex3];
  const int out_stride4 = out_strides[kIndex4];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  const int output3 = output_shape[kIndex3];
  const int output4 = output_shape[kIndex4];
  const int output5 = output_shape[kIndex5];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < static_cast<uint32_t>(output2); ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < static_cast<uint32_t>(output3); ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < static_cast<uint32_t>(output4); ++n) {
            size_t out_stride4_n = n * out_stride4;
            size_t stride4_n = n * stride4;
            for (size_t g = 0; g < static_cast<uint32_t>(output5); ++g) {
              out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + out_stride4_n + g] =
                in_data[stride0_i + stride1_j + stride2_k + stride3_m + stride4_n + g * stride5];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDim7(const T *in_data, T *out_data, const int *strides,
                                                   const int *out_strides, const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int stride3 = strides[perm[kIndex3]];
  const int stride4 = strides[perm[kIndex4]];
  const int stride5 = strides[perm[kIndex5]];
  const int stride6 = strides[perm[kIndex6]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int out_stride2 = out_strides[kIndex2];
  const int out_stride3 = out_strides[kIndex3];
  const int out_stride4 = out_strides[kIndex4];
  const int out_stride5 = out_strides[kIndex5];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  const int output3 = output_shape[kIndex3];
  const int output4 = output_shape[kIndex4];
  const int output5 = output_shape[kIndex5];
  const int output6 = output_shape[kIndex6];
  for (size_t i = 0; i < static_cast<uint32_t>(output0); ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < static_cast<uint32_t>(output1); ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < static_cast<uint32_t>(output2); ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < static_cast<uint32_t>(output3); ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < static_cast<uint32_t>(output4); ++n) {
            size_t out_stride4_n = n * out_stride4;
            size_t stride4_n = n * stride4;
            for (size_t g = 0; g < static_cast<uint32_t>(output5); ++g) {
              size_t out_stride5_g = g * out_stride5;
              size_t stride5_g = g * stride5;
              for (size_t s = 0; s < static_cast<uint32_t>(output6); ++s) {
                out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + out_stride4_n + out_stride5_g +
                         s] =
                  in_data[stride0_i + stride1_j + stride2_k + stride3_m + stride4_n + stride5_g + s * stride6];
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void ConjugateTransposeCpuKernelMod::ParallelRun(const T *input_addr, T *output_addr, const int *output_shape,
                                                 size_t count, const TransposeParameter *transpose_param) {
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  const size_t thread_num =
    count < block_size * max_thread_num ? FloatToSize(std::ceil(count / block_size)) : max_thread_num;
  std::vector<common::Task> tasks;
  for (int task_id = 0; task_id < SizeToInt(thread_num); ++task_id) {
    auto task = [this, &input_addr, &output_addr, &output_shape, &transpose_param, task_id, thread_num]() {
      TransposeDims(input_addr, output_addr, output_shape, transpose_param, task_id, SizeToInt(thread_num));
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}

template <typename T>
void ConjugateTransposeCpuKernelMod::TransposeDims(const T *in_data, T *out_data, const int *output_shape,
                                                   const TransposeParameter *transpose_param, int task_id,
                                                   int thread_num) {
  NNACL_CHECK_NULL_RETURN_VOID(in_data);
  NNACL_CHECK_NULL_RETURN_VOID(out_data);
  NNACL_CHECK_NULL_RETURN_VOID(output_shape);
  NNACL_CHECK_NULL_RETURN_VOID(transpose_param);
  NNACL_CHECK_ZERO_RETURN(thread_num);
  const int *perm = transpose_param->perm_;
  const int *strides = transpose_param->strides_;
  const int *out_strides = transpose_param->out_strides_;
  int num_axes = transpose_param->num_axes_;
  size_t data_size_ = (*out_strides) * output_shape[0];
  size_t offset_size = UP_DIV(data_size_, thread_num);
  size_t task_offset = offset_size * static_cast<size_t>(task_id);
  int count = static_cast<int32_t>(data_size_) - static_cast<int32_t>(task_offset);
  if (count <= 0) {
    return;
  }
  count = MSMIN(offset_size, static_cast<uint32_t>(count));
  for (int idx = static_cast<int32_t>(task_offset); static_cast<uint32_t>(idx) < task_offset + count; ++idx) {
    int pos = idx;
    int output_idx = 0;
    int input_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      NNACL_CHECK_ZERO_RETURN(*(out_strides + i));
      int position = pos / *(out_strides + i);
      int out_stride = i < num_axes - 1 ? out_strides[i] : 1;
      output_idx += (position * out_stride);
      input_idx += (position * strides[perm[i]]);
      pos -= position * (*(out_strides + i));
    }
    out_data[output_idx] = in_data[input_idx];
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ConjugateTranspose, ConjugateTransposeCpuKernelMod);
}  // namespace conjugate_transpose_cpu
}  // namespace kernel
}  // namespace mindspore
