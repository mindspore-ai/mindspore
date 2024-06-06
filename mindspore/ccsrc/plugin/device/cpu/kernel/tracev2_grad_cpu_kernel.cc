/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/tracev2_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 5;
constexpr size_t kOutputNum = 1;
constexpr size_t kIndexDout = 0;
constexpr size_t kIndexInShape = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kIndexAxis1 = 3;
constexpr size_t kIndexAxis2 = 4;
constexpr size_t kIndexDin = 0;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool TraceV2GradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  return true;
}

int TraceV2GradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[kIndexInShape]->GetValueWithCheck<std::vector<int64_t>>();
  int64_t in_rank = static_cast<int64_t>(input_shape.size());
  din_size_ = SizeOf(input_shape);
  offset_ = inputs[kIndexOffset]->GetValueWithCheck<int64_t>();
  int64_t axis1 = inputs[kIndexAxis1]->GetValueWithCheck<int64_t>();
  int64_t axis2 = inputs[kIndexAxis2]->GetValueWithCheck<int64_t>();
  axis1 = axis1 < 0 ? axis1 + in_rank : axis1;
  axis2 = axis2 < 0 ? axis2 + in_rank : axis2;
  mat_size_ = input_shape[axis1] * input_shape[axis2];
  mat_row_size_ = input_shape[axis2];
  mat_col_size_ = input_shape[axis1];

  std::vector<int64_t> trans_input_shape;
  std::vector<int64_t> rev_perm_vec;
  batch_size_ = 1;

  for (int64_t i = 0; i < in_rank; i++) {
    if (i != axis1 && i != axis2) {
      trans_input_shape.emplace_back(input_shape[i]);
      rev_perm_vec.emplace_back(i);
      batch_size_ *= input_shape[i];
    }
  }
  trans_input_shape.emplace_back(input_shape[axis1]);
  trans_input_shape.emplace_back(input_shape[axis2]);
  rev_perm_vec.emplace_back(axis1);
  rev_perm_vec.emplace_back(axis2);

  TransposeIterator rev_iter(trans_input_shape, LongVecToSizeVec(rev_perm_vec), input_shape);
  rev_tanspose_index_.clear();
  rev_iter.SetPos(0);
  for (size_t i = 0; i < din_size_; i++) {
    rev_tanspose_index_.emplace_back(rev_iter.GetPos());
    rev_iter.GenNextPos();
  }

  return KRET_OK;
}

bool TraceV2GradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto dout_type = inputs.at(kIndex0)->dtype_id();
  switch (dout_type) {
    case kNumberTypeInt8:
      LaunchKernel<int8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      LaunchKernel<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      LaunchKernel<int16_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      LaunchKernel<uint16_t>(inputs, outputs);
      break;
    case kNumberTypeFloat16:
      LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeUInt32:
      LaunchKernel<uint32_t>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    case kNumberTypeUInt64:
      LaunchKernel<uint64_t>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double>(inputs, outputs);
      break;
    case kNumberTypeComplex64:
      LaunchKernel<complex64>(inputs, outputs);
      break;
    case kNumberTypeComplex128:
      LaunchKernel<complex128>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "Trace Grad Unsupported input data type.";
  }
  return true;
}

template <typename T>
void TraceV2GradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  T *dout_addr = reinterpret_cast<T *>(inputs[kIndexDout]->device_ptr());
  MS_EXCEPTION_IF_NULL(dout_addr);
  T *din_addr = reinterpret_cast<T *>(outputs[kIndexDin]->device_ptr());
  MS_EXCEPTION_IF_NULL(din_addr);

  auto ret = memset_s(din_addr, din_size_ * sizeof(T), 0, din_size_ * sizeof(T));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For 'TraceV2', memset_s failed, ret=" << ret;
  }

  for (int64_t i = 0; i < batch_size_; i++) {
    int64_t row_idx;
    int64_t col_idx;
    if (offset_ > 0) {
      row_idx = 0;
      col_idx = offset_;
    } else {
      col_idx = 0;
      row_idx = -offset_;
    }
    while (row_idx < mat_col_size_ && col_idx < mat_row_size_) {
      int64_t idx = row_idx * mat_row_size_ + col_idx + i * mat_size_;
      din_addr[rev_tanspose_index_[idx]] = dout_addr[i];
      row_idx++;
      col_idx++;
    }
  }
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TraceV2Grad, TraceV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
