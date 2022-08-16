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

#include "extendrt/kernel/cpu/transpose_kernel_mod.h"
#include <vector>
#include <memory>
#include "plugin/factory/ms_factory.h"
#include "include/api/status.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore::kernel {
namespace {
constexpr size_t kTransposeInputsNum = 2;
constexpr size_t kTransposeOutputsNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
// kMaxTransposeSerialSize = 64 * 3 * 512 * 512
constexpr size_t kMaxTransposeSerialSize = 50331648;
}  // namespace

bool TransposeKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTransposeOutputsNum, kernel_name_);
  launch_func_(this, inputs, outputs);
  return true;
}

int TransposeKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  return kSuccess;
}

bool TransposeKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTransposeOutputsNum, kernel_name_);
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  auto address_ptr = inputs[kIndex1]->GetData();
  if (address_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Address ptr is nullptr.";
  }
  int *addr = static_cast<int *>(address_ptr->addr);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Cast addr failed.";
  }
  std::vector<int64_t> perm;
  for (size_t i = 0; i < (address_ptr->size) / sizeof(int); ++i) {
    perm.emplace_back(static_cast<int64_t>(addr[i]));
  }
  for (auto p : perm) {
    p = (p >= 0) ? p : (perm.size() + p);
    if (p < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm.size() << ", "
                        << (perm.size() - 1) << "], but got " << perm;
    }
    axes_.emplace_back(p);
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  if (axes_.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(EXCEPTION) << "Transpose support max dimension is " << MAX_TRANSPOSE_DIM_SIZE << "D, but got "
                      << axes_.size() << "D.";
  }
  for (size_t i = 0; i < axes_.size(); ++i) {
    transpose_param_.perm_[i] = SizeToInt(axes_[i]);
  }
  size_t num_axes = input_shape_.size();
  transpose_param_.perm_size_ = axes_.size();
  transpose_param_.num_axes_ = SizeToInt(num_axes);
  transpose_param_.strides_[num_axes - 1] = 1;
  transpose_param_.out_strides_[num_axes - 1] = 1;
  for (size_t i = num_axes - 1; i >= 1; i--) {
    transpose_param_.strides_[i - 1] = input_shape_[i] * transpose_param_.strides_[i];
    transpose_param_.out_strides_[i - 1] = output_shape_[i] * transpose_param_.out_strides_[i];
  }
  launch_map_[kNumberTypeBool] = &TransposeKernelMod::LaunchKernel<bool>;
  launch_map_[kNumberTypeInt8] = &TransposeKernelMod::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TransposeKernelMod::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TransposeKernelMod::LaunchKernel<int32_t>;
  launch_map_[kNumberTypeInt64] = &TransposeKernelMod::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TransposeKernelMod::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TransposeKernelMod::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TransposeKernelMod::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TransposeKernelMod::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat16] = &TransposeKernelMod::LaunchKernel<float16>;
  launch_map_[kNumberTypeFloat32] = &TransposeKernelMod::LaunchKernel<float>;
  launch_map_[kNumberTypeFloat64] = &TransposeKernelMod::LaunchKernel<double>;
  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  free(address_ptr->addr);
  inputs[kIndex1]->GetData()->addr = nullptr;
  inputs[kIndex1]->GetData()->size = 0;
  return true;
}

template <typename T>
void TransposeKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  transpose_param_.data_num_ = SizeToInt(inputs[0]->size / sizeof(T));
  int output_shape[SizeToInt(output_shape_.size())];
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_shape[i] = output_shape_[i];
  }
  bool res{static_cast<bool>(NNACL_OK)};
  res = DoTranspose(input_addr, output_addr, output_shape, &transpose_param_);
  if (res != static_cast<bool>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "Transpose run failed.";
  }
}

template <typename T>
int TransposeKernelMod::DoTranspose(const T *in_data, T *out_data, const int *output_shape,
                                    const TransposeParameter *transpose_param) {
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  NNACL_CHECK_NULL_RETURN_ERR(output_shape);
  NNACL_CHECK_NULL_RETURN_ERR(transpose_param);
  const int *perm = transpose_param->perm_;
  const int *strides = transpose_param->strides_;
  const int *out_strides = transpose_param->out_strides_;
  int data_size = transpose_param->data_num_ * sizeof(T);
  int num_axes = transpose_param->num_axes_;
  bool needTranspose = false;
  for (size_t i = 1; i < (unsigned int)num_axes; ++i) {
    if (perm[i] - perm[i - 1] != 1) {
      needTranspose = true;
      break;
    }
  }
  if (!needTranspose) {
    (void)memcpy(out_data, in_data, data_size);
    return NNACL_OK;
  }
  for (size_t i = 0; i < (unsigned int)num_axes; ++i) {
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
void TransposeKernelMod::TransposeDim2(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * output1;
    size_t stride0_i = i * 1 * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      out_data[out_stride0_i + j] = in_data[stride0_i + j * stride1];
    }
  }
}

template <typename T>
void TransposeKernelMod::TransposeDim3(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[kIndex0]];
  const int stride1 = strides[perm[kIndex1]];
  const int stride2 = strides[perm[kIndex2]];
  const int out_stride0 = out_strides[kIndex0];
  const int out_stride1 = out_strides[kIndex1];
  const int output0 = output_shape[kIndex0];
  const int output1 = output_shape[kIndex1];
  const int output2 = output_shape[kIndex2];
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < (unsigned int)output2; ++k) {
        out_data[out_stride0_i + out_stride1_j + k] = in_data[stride0_i + stride1_j + k * stride2];
      }
    }
  }
}

template <typename T>
void TransposeKernelMod::TransposeDim4(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
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
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < (unsigned int)output2; ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < (unsigned int)output3; ++m) {
          out_data[out_stride0_i + out_stride1_j + out_stride2_k + m] =
            in_data[stride0_i + stride1_j + stride2_k + m * stride3];
        }
      }
    }
  }
}

template <typename T>
void TransposeKernelMod::TransposeDim5(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
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
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < (unsigned int)output2; ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < (unsigned int)output3; ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < (unsigned int)output4; ++n) {
            out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + n] =
              in_data[stride0_i + stride1_j + stride2_k + stride3_m + n * stride4];
          }
        }
      }
    }
  }
}

template <typename T>
void TransposeKernelMod::TransposeDim6(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
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
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < (unsigned int)output2; ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < (unsigned int)output3; ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < (unsigned int)output4; ++n) {
            size_t out_stride4_n = n * out_stride4;
            size_t stride4_n = n * stride4;
            for (size_t g = 0; g < (unsigned int)output5; ++g) {
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
void TransposeKernelMod::TransposeDim7(const T *in_data, T *out_data, const int *strides, const int *out_strides,
                                       const int *perm, const int *output_shape) {
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
  for (size_t i = 0; i < (unsigned int)output0; ++i) {
    size_t out_stride0_i = i * out_stride0;
    size_t stride0_i = i * stride0;
    for (size_t j = 0; j < (unsigned int)output1; ++j) {
      size_t out_stride1_j = j * out_stride1;
      size_t stride1_j = j * stride1;
      for (size_t k = 0; k < (unsigned int)output2; ++k) {
        size_t out_stride2_k = k * out_stride2;
        size_t stride2_k = k * stride2;
        for (size_t m = 0; m < (unsigned int)output3; ++m) {
          size_t out_stride3_m = m * out_stride3;
          size_t stride3_m = m * stride3;
          for (size_t n = 0; n < (unsigned int)output4; ++n) {
            size_t out_stride4_n = n * out_stride4;
            size_t stride4_n = n * stride4;
            for (size_t g = 0; g < (unsigned int)output5; ++g) {
              size_t out_stride5_g = g * out_stride5;
              size_t stride5_g = g * stride5;
              for (size_t s = 0; s < (unsigned int)output6; ++s) {
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
void TransposeKernelMod::TransposeDims(const T *in_data, T *out_data, const int *output_shape,
                                       const TransposeParameter *transpose_param, int task_id, int thread_num) {
  NNACL_CHECK_NULL_RETURN_VOID(in_data);
  NNACL_CHECK_NULL_RETURN_VOID(out_data);
  NNACL_CHECK_NULL_RETURN_VOID(output_shape);
  NNACL_CHECK_NULL_RETURN_VOID(transpose_param);
  NNACL_CHECK_ZERO_RETURN(thread_num);
  const int *perm = transpose_param->perm_;
  const int *strides = transpose_param->strides_;
  const int *out_strides = transpose_param->out_strides_;
  int num_axes = transpose_param->num_axes_;
  size_t data_size = (*out_strides) * output_shape[0];
  size_t offset_size = UP_DIV(data_size, thread_num);
  size_t task_offset = offset_size * task_id;
  int count = data_size - task_offset;
  if (count <= 0) {
    return;
  }
  count = MSMIN(offset_size, (unsigned int)count);
  for (int idx = task_offset; (unsigned int)idx < task_offset + count; ++idx) {
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

MS_KERNEL_FACTORY_REG_BY_CREATOR(KernelMod, Transpose,
                                 []() { return std::make_shared<TransposeKernelMod>(prim::kPrimTranspose->name()); });
}  // namespace mindspore::kernel
