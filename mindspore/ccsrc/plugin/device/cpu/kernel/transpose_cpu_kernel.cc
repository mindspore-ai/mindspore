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

#include <complex>
#include <vector>
#include <algorithm>
#include "nnacl/errorcode.h"
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/kernel/transpose_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTransposeInputsNum = 1;
constexpr size_t kDynamicPermInputNum = 2;
constexpr size_t kTransposeOutputsNum = 1;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
// kMaxTransposeSerialSize = 64 * 3 * 512 * 512
constexpr size_t kMaxTransposeSerialSize = 50331648;
}  // namespace

void TransposeFwdCpuKernelMod::CheckPermValue() {
  for (auto &p : perm_) {
    p = (p >= 0) ? p : (SizeToLong(perm_.size()) + p);
    if (p < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm value must be in [-" << perm_.size() << ", "
                        << (perm_.size() - 1) << "], but got " << p;
    }
  }
  if (!IsDynamicRank(input_shape_) && perm_.size() != input_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the perm's size must be equal to input_shape's size, but got "
                      << perm_.size() << " vs " << input_shape_.size();
  }

  if (perm_.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(EXCEPTION) << "Transpose support max dimension is " << MAX_TRANSPOSE_DIM_SIZE << "D, but got "
                      << perm_.size() << "D.";
  }
}

template <typename T>
void TransposeFwdCpuKernelMod::InitPerm(const std::vector<kernel::AddressPtr> &inputs) {
  auto cnode = cnode_ptr_.lock();
  auto perm_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex1);
  auto perm_ptr = static_cast<T *>(inputs[kIndex1]->addr);
  std::vector<T> perm{perm_ptr, perm_ptr + perm_shape[0]};
  (void)std::transform(perm.begin(), perm.end(), std::back_inserter(perm_),
                       [](const T &value) { return static_cast<int64_t>(value); });
  CheckPermValue();
}

void TransposeFwdCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  cnode_ptr_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1 && input_num != kDynamicPermInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1 or " << kDynamicPermInputNum
                      << ", but got " << input_num;
  }
  if (!IsDynamicRank(output_shape_) && output_shape_.size() != input_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the output_shape's size must be equal to input_shape's size, but got "
                      << output_shape_.size() << " vs " << input_shape_.size();
  }
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  num_axes_ = input_shape_.size();
  if (num_axes_ == 0) {
    MS_LOG(EXCEPTION) << "Transpose's input shape is empty.";
  }
  strides_.resize(num_axes_);
  out_strides_.resize(num_axes_);
  strides_[num_axes_ - 1] = 1LL;
  out_strides_[num_axes_ - 1] = 1LL;
  for (size_t i = num_axes_ - 1; i >= 1; i--) {
    strides_[i - 1] = input_shape_[i] * strides_[i];
    out_strides_[i - 1] = output_shape_[i] * out_strides_[i];
  }
  launch_map_[kNumberTypeBool] = &TransposeFwdCpuKernelMod::LaunchKernel<bool>;
  launch_map_[kNumberTypeInt8] = &TransposeFwdCpuKernelMod::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TransposeFwdCpuKernelMod::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TransposeFwdCpuKernelMod::LaunchKernel<int32_t>;
  launch_map_[kNumberTypeInt64] = &TransposeFwdCpuKernelMod::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TransposeFwdCpuKernelMod::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TransposeFwdCpuKernelMod::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TransposeFwdCpuKernelMod::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TransposeFwdCpuKernelMod::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat16] = &TransposeFwdCpuKernelMod::LaunchKernel<float16>;
  launch_map_[kNumberTypeFloat32] = &TransposeFwdCpuKernelMod::LaunchKernel<float>;
  launch_map_[kNumberTypeFloat64] = &TransposeFwdCpuKernelMod::LaunchKernel<double>;
  launch_map_[kNumberTypeComplex64] = &TransposeFwdCpuKernelMod::LaunchKernel<complex64>;
  launch_map_[kNumberTypeComplex128] = &TransposeFwdCpuKernelMod::LaunchKernel<complex128>;
  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  if (input_num == kDynamicPermInputNum) {
    perm_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex1);
    return;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto value_ptr = prim->GetAttr("perm");
  if (value_ptr->isa<tensor::Tensor>()) {
    perm_ = CheckAndConvertUtils::CheckTensorIntValue("perm", value_ptr, kernel_name_);
  } else {
    perm_ = CheckAndConvertUtils::CheckIntOrTupleInt("perm", value_ptr, kernel_name_);
  }
  CheckPermValue();
}

bool TransposeFwdCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTransposeOutputsNum, kernel_name_);
  if (inputs.size() == kDynamicPermInputNum) {
    if (perm_type_ == kNumberTypeInt32) {
      InitPerm<int32_t>(inputs);
    } else {
      InitPerm<int64_t>(inputs);
    }
  }
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TransposeFwdCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
  const auto *input_addr = static_cast<T *>(inputs[0]->addr);
  auto *output_addr = static_cast<T *>(outputs[0]->addr);
  data_num_ = inputs[0]->size / sizeof(T);
  size_t data_count = (inputs[0]->size) / sizeof(T);
  if (perm_.size() > kIndex7 || data_count >= kMaxTransposeSerialSize) {
    ParallelRun(input_addr, output_addr, data_count);
    return;
  }

  ErrorCodeCommonEnum res = DoTranspose(input_addr, output_addr);
  if (res != NNACL_OK) {
    MS_LOG(EXCEPTION) << "Transpose run failed.";
  }
}

template <typename T>
ErrorCodeCommonEnum TransposeFwdCpuKernelMod::DoTranspose(const T *in_data, T *out_data) const {
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  bool needTranspose = false;
  for (size_t i = 1; i < num_axes_; ++i) {
    if (perm_[i] - perm_[i - 1] != 1) {
      needTranspose = true;
      break;
    }
  }
  if (!needTranspose) {
    (void)std::copy(in_data, in_data + data_num_, out_data);
    return NNACL_OK;
  }
  for (size_t i = 0; i < num_axes_; ++i) {
    if (perm_[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  if (num_axes_ == kIndex2) {
    TransposeDim2(in_data, out_data);
  } else if (num_axes_ == kIndex3) {
    TransposeDim3(in_data, out_data);
  } else if (num_axes_ == kIndex4) {
    TransposeDim4(in_data, out_data);
  } else if (num_axes_ == kIndex5) {
    TransposeDim5(in_data, out_data);
  } else if (num_axes_ == kIndex6) {
    TransposeDim6(in_data, out_data);
  } else if (num_axes_ == kIndex7) {
    TransposeDim7(in_data, out_data);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDim2(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * output1;
    int64_t stride0_i = i * 1 * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      out_data[out_stride0_i + j] = in_data[stride0_i + j * stride1];
    }
  }
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDim3(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto stride2 = strides_[perm_[kIndex2]];
  const auto out_stride0 = out_strides_[kIndex0];
  const auto out_stride1 = out_strides_[kIndex1];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  const auto output2 = output_shape_[kIndex2];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * out_stride0;
    int64_t stride0_i = i * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_stride1_j = j * out_stride1;
      int64_t stride1_j = j * stride1;
      for (int64_t k = 0; k < output2; ++k) {
        out_data[out_stride0_i + out_stride1_j + k] = in_data[stride0_i + stride1_j + k * stride2];
      }
    }
  }
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDim4(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto stride2 = strides_[perm_[kIndex2]];
  const auto stride3 = strides_[perm_[kIndex3]];
  const auto out_stride0 = out_strides_[kIndex0];
  const auto out_stride1 = out_strides_[kIndex1];
  const auto out_stride2 = out_strides_[kIndex2];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  const auto output2 = output_shape_[kIndex2];
  const auto output3 = output_shape_[kIndex3];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * out_stride0;
    int64_t stride0_i = i * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_stride1_j = j * out_stride1;
      int64_t stride1_j = j * stride1;
      for (int64_t k = 0; k < output2; ++k) {
        int64_t out_stride2_k = k * out_stride2;
        int64_t stride2_k = k * stride2;
        for (int64_t m = 0; m < output3; ++m) {
          out_data[out_stride0_i + out_stride1_j + out_stride2_k + m] =
            in_data[stride0_i + stride1_j + stride2_k + m * stride3];
        }
      }
    }
  }
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDim5(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto stride2 = strides_[perm_[kIndex2]];
  const auto stride3 = strides_[perm_[kIndex3]];
  const auto stride4 = strides_[perm_[kIndex4]];
  const auto out_stride0 = out_strides_[kIndex0];
  const auto out_stride1 = out_strides_[kIndex1];
  const auto out_stride2 = out_strides_[kIndex2];
  const auto out_stride3 = out_strides_[kIndex3];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  const auto output2 = output_shape_[kIndex2];
  const auto output3 = output_shape_[kIndex3];
  const auto output4 = output_shape_[kIndex4];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * out_stride0;
    int64_t stride0_i = i * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_stride1_j = j * out_stride1;
      int64_t stride1_j = j * stride1;
      for (int64_t k = 0; k < output2; ++k) {
        int64_t out_stride2_k = k * out_stride2;
        int64_t stride2_k = k * stride2;
        for (int64_t m = 0; m < output3; ++m) {
          int64_t out_stride3_m = m * out_stride3;
          int64_t stride3_m = m * stride3;
          for (int64_t n = 0; n < output4; ++n) {
            out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + n] =
              in_data[stride0_i + stride1_j + stride2_k + stride3_m + n * stride4];
          }
        }
      }
    }
  }
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDim6(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto stride2 = strides_[perm_[kIndex2]];
  const auto stride3 = strides_[perm_[kIndex3]];
  const auto stride4 = strides_[perm_[kIndex4]];
  const auto stride5 = strides_[perm_[kIndex5]];
  const auto out_stride0 = out_strides_[kIndex0];
  const auto out_stride1 = out_strides_[kIndex1];
  const auto out_stride2 = out_strides_[kIndex2];
  const auto out_stride3 = out_strides_[kIndex3];
  const auto out_stride4 = out_strides_[kIndex4];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  const auto output2 = output_shape_[kIndex2];
  const auto output3 = output_shape_[kIndex3];
  const auto output4 = output_shape_[kIndex4];
  const auto output5 = output_shape_[kIndex5];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * out_stride0;
    int64_t stride0_i = i * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_stride1_j = j * out_stride1;
      int64_t stride1_j = j * stride1;
      for (int64_t k = 0; k < output2; ++k) {
        int64_t out_stride2_k = k * out_stride2;
        int64_t stride2_k = k * stride2;
        for (int64_t m = 0; m < output3; ++m) {
          int64_t out_stride3_m = m * out_stride3;
          int64_t stride3_m = m * stride3;
          for (int64_t n = 0; n < output4; ++n) {
            int64_t out_stride4_n = n * out_stride4;
            int64_t stride4_n = n * stride4;
            for (int64_t g = 0; g < output5; ++g) {
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
void TransposeFwdCpuKernelMod::TransposeDim7(const T *in_data, T *out_data) const {
  const auto stride0 = strides_[perm_[kIndex0]];
  const auto stride1 = strides_[perm_[kIndex1]];
  const auto stride2 = strides_[perm_[kIndex2]];
  const auto stride3 = strides_[perm_[kIndex3]];
  const auto stride4 = strides_[perm_[kIndex4]];
  const auto stride5 = strides_[perm_[kIndex5]];
  const auto stride6 = strides_[perm_[kIndex6]];
  const auto out_stride0 = out_strides_[kIndex0];
  const auto out_stride1 = out_strides_[kIndex1];
  const auto out_stride2 = out_strides_[kIndex2];
  const auto out_stride3 = out_strides_[kIndex3];
  const auto out_stride4 = out_strides_[kIndex4];
  const auto out_stride5 = out_strides_[kIndex5];
  const auto output0 = output_shape_[kIndex0];
  const auto output1 = output_shape_[kIndex1];
  const auto output2 = output_shape_[kIndex2];
  const auto output3 = output_shape_[kIndex3];
  const auto output4 = output_shape_[kIndex4];
  const auto output5 = output_shape_[kIndex5];
  const auto output6 = output_shape_[kIndex6];
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_stride0_i = i * out_stride0;
    int64_t stride0_i = i * stride0;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_stride1_j = j * out_stride1;
      int64_t stride1_j = j * stride1;
      for (int64_t k = 0; k < output2; ++k) {
        int64_t out_stride2_k = k * out_stride2;
        int64_t stride2_k = k * stride2;
        for (int64_t m = 0; m < output3; ++m) {
          int64_t out_stride3_m = m * out_stride3;
          int64_t stride3_m = m * stride3;
          for (int64_t n = 0; n < output4; ++n) {
            int64_t out_stride4_n = n * out_stride4;
            int64_t stride4_n = n * stride4;
            for (int64_t g = 0; g < output5; ++g) {
              int64_t out_stride5_g = g * out_stride5;
              int64_t stride5_g = g * stride5;
              for (int64_t s = 0; s < output6; ++s) {
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
void TransposeFwdCpuKernelMod::ParallelRun(const T *input_addr, T *output_addr, size_t count) const {
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  const int64_t thread_num =
    SizeToLong(count < block_size * max_thread_num ? FloatToSize(std::ceil(count / block_size)) : max_thread_num);
  std::vector<common::Task> tasks;
  for (int64_t task_id = 0; task_id < thread_num; ++task_id) {
    auto task = [this, &input_addr, &output_addr, task_id, thread_num]() {
      TransposeDims(input_addr, output_addr, task_id, thread_num);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
  }
  ParallelLaunch(tasks);
}

template <typename T>
void TransposeFwdCpuKernelMod::TransposeDims(const T *in_data, T *out_data, int64_t task_id, int64_t thread_num) const {
  NNACL_CHECK_NULL_RETURN_VOID(in_data);
  NNACL_CHECK_NULL_RETURN_VOID(out_data);
  NNACL_CHECK_ZERO_RETURN(thread_num);
  auto data_size = out_strides_[0] * output_shape_[0];
  auto offset_size = UP_DIV(data_size, thread_num);
  auto task_offset = offset_size * task_id;
  auto count = data_size - task_offset;
  if (count <= 0) {
    return;
  }
  count = MSMIN(offset_size, count);
  for (int64_t idx = task_offset; idx < task_offset + count; ++idx) {
    int64_t pos = idx;
    int64_t output_idx = 0;
    int64_t input_idx = 0;
    for (size_t i = 0; i < num_axes_; ++i) {
      NNACL_CHECK_ZERO_RETURN(out_strides_[i]);
      int64_t position = pos / out_strides_[i];
      int64_t out_stride = i + 1 < num_axes_ ? out_strides_[i] : 1LL;
      output_idx += (position * out_stride);
      input_idx += (position * strides_[perm_[i]]);
      pos -= position * out_strides_[i];
    }
    out_data[output_idx] = in_data[input_idx];
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Transpose, TransposeFwdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
