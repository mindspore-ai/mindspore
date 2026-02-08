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

#include "kernel/cpu/native/slice_cpu_kernel.h"
#include <complex>
#include <algorithm>
#include <unordered_map>
#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace slice_cpu {
namespace {
constexpr size_t kSliceInputsNum = 3;
constexpr size_t kSliceOutputsNum = 1;
constexpr size_t kSliceInputIndex2 = 2;
constexpr size_t kSliceTwoDims = 2;
constexpr char kKernelName[] = "Slice";
}  // namespace

int NormalizeBeginPos(int begin_pos, int dim_len) {
  if (begin_pos < 0) {
    int normal_pos = begin_pos + dim_len;
    return std::max(normal_pos, 0);
  }
  return std::min(begin_pos, dim_len - 1);
}

bool SliceCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  static const std::unordered_map<TypeId, int> type_size_map = {{kNumberTypeBool, sizeof(bool)},
                                                                {kNumberTypeInt8, sizeof(int8_t)},
                                                                {kNumberTypeInt16, sizeof(int16_t)},
                                                                {kNumberTypeInt32, sizeof(int32_t)},
                                                                {kNumberTypeInt64, sizeof(int64_t)},
                                                                {kNumberTypeUInt8, sizeof(uint8_t)},
                                                                {kNumberTypeUInt16, sizeof(uint16_t)},
                                                                {kNumberTypeUInt32, sizeof(uint32_t)},
                                                                {kNumberTypeUInt64, sizeof(uint64_t)},
                                                                {kNumberTypeFloat32, sizeof(float)},
                                                                {kNumberTypeFloat64, sizeof(double)},
                                                                {kNumberTypeFloat16, sizeof(float16)},
                                                                {kNumberTypeComplex64, sizeof(std::complex<float>)},
                                                                {kNumberTypeComplex128, sizeof(std::complex<double>)},
                                                                {kNumberTypeBFloat16, sizeof(bfloat16)}};

  size_t input_num = inputs.size();
  if (input_num != kSliceInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "input size should be " << kSliceInputsNum;
  }

  TypeId dtype = inputs[kIndex0]->dtype_id();
  auto size_pair = type_size_map.find(dtype);
  if (size_pair == type_size_map.end()) {
    MS_LOG(EXCEPTION) << "Slice supports type in type_size_map, but got " << TypeIdToType(dtype)->ToString();
  }
  data_size_ = size_pair->second;

  MS_EXCEPTION_IF_CHECK_FAIL(inputs.at(1)->dtype_id() == inputs[kSliceInputIndex2]->dtype_id(),
                             "Begin and size dtype should be same.");
  param_dtype_ = inputs.at(1)->dtype_id();

  return true;
}

void SliceCpuKernelMod::InitSliceParam(const ShapeVector &input_shape, const std::vector<int64_t> &begin,
                                       const std::vector<int64_t> &size) {
  origin_dim_size_ = input_shape.size();
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i < input_shape.size()) {
      int dim_len = LongToInt(input_shape[i]);
      int begin_pos = LongToInt(begin[i]);
      int slice_size = LongToInt(size[i]);
      if (slice_size == -1) {
        slice_size = dim_len - begin_pos;
      }
      if (slice_size < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the each dimension of 'size' must be greater than or equal to 0 "
                             "or be equal to -1, but got slice size "
                          << slice_size;
      }
      slice_param_.shape_[i] = dim_len;
      slice_param_.size_[i] = slice_size;
      slice_param_.begin_[i] = NormalizeBeginPos(begin_pos, dim_len);
      int end = slice_param_.begin_[i] + slice_param_.size_[i];
      slice_param_.end_[i] = std::min(end, dim_len);
    } else {
      slice_param_.shape_[i] = 1;
      slice_param_.begin_[i] = 0;
      slice_param_.size_[i] = 1;
      slice_param_.end_[i] = 1;
    }
  }
  slice_param_.param_length_ = DIMENSION_8D;
}

int SliceCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  is_got_value_ = false;
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  size_t input_num = inputs.size();
  if (input_num != kSliceInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "input size should be " << kSliceInputsNum;
  }

  auto input_shape = inputs[0]->GetShapeVector();
  if (input_shape.size() > DIMENSION_8D || input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input tensor must be in range [1D, 8D], but got " << input_shape.size()
                      << "D.";
  }
  auto input_size = IntToLong(data_size_) * SizeToLong(SizeOf(input_shape));
  if (input_size > INT_MAX) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input size can not larger than " << INT_MAX
                      << "(INT_MAX) bytes, but got " << input_size;
  }

  std::vector<int64_t> begin;
  std::vector<int64_t> size;
  if (param_dtype_ == kNumberTypeInt32) {
    auto begin_int32 = inputs[kIndex1]->GetValueWithCheck<std::vector<int32_t>>();
    auto size_int32 = inputs[kIndex2]->GetValueWithCheck<std::vector<int32_t>>();
    if (begin_int32.size() == input_shape.size() && size_int32.size() == input_shape.size()) {
      for (size_t i = 0; i < input_shape.size(); i++) {
        begin.push_back(begin_int32.at(i));
        size.push_back(size_int32.at(i));
      }
    }
  } else if (param_dtype_ == kNumberTypeInt64) {
    begin = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
    size = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    MS_LOG(ERROR) << "Invalid dtype: " << TypeIdLabel(param_dtype_);
    return false;
  }

  if (!begin.empty() && !size.empty()) {
    if (begin.size() != input_shape.size() || size.size() != input_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the lengths of 'begin' and 'size' must be equal to "
                       "the dimension of input tensor, but got the length of 'begin' "
                    << begin.size() << ", the length of 'size' " << size.size() << "and the dimension of input tensor "
                    << input_shape.size();
      return KRET_RESIZE_FAILED;
    }
    InitSliceParam(input_shape, begin, size);
    is_got_value_ = true;
  } else {
    input_shape_ = inputs[kIndex0]->GetShapeVector();
    begin_shape_ = inputs[kIndex1]->GetShapeVector();
    size_shape_ = inputs[kSliceInputIndex2]->GetShapeVector();
    is_got_value_ = false;
  }
  return KRET_OK;
}

void SliceSimpleDim2(const int8_t *input, int8_t *output, const SliceStruct *param, int data_size, size_t row_size) {
  size_t copy_size = IntToSize(data_size * param->size_[1]);
  for (size_t i = 0; i < row_size; ++i) {
    auto dst = output + data_size * param->size_[1] * SizeToInt(i);
    auto src = input + data_size * (param->shape_[1] * SizeToInt(i) + param->begin_[1]);
    auto ret = memcpy_s(dst, copy_size, src, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kKernelName << "', memcpy failed. Error no: " << ret;
    }
  }
}

bool SliceCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  if (inputs.size() != kSliceInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kSliceInputsNum
                      << ", but got " << inputs.size() << " input(s).";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSliceOutputsNum, kernel_name_);

  if (!is_got_value_) {
    // Get begin and size value.
    auto input_shape = input_shape_;
    auto begin_shape = begin_shape_;
    auto size_shape = size_shape_;
    if (begin_shape.size() != 1 || size_shape.size() != 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the dimensions of 'begin' and 'size' must be 1, but got the dimension of 'begin': "
                    << begin_shape.size() << " and the dimension of 'size': " << size_shape.size();
      return false;
    }
    if (begin_shape[0] != SizeToLong(input_shape.size()) || size_shape[0] != SizeToLong(input_shape.size())) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the lengths of 'begin' and 'size' must be equal to "
                       "the dimension of input tensor, but got the length of 'begin' "
                    << begin_shape[0] << ", the length of 'size' " << size_shape[0]
                    << "and the dimension of input tensor " << input_shape.size();
      return false;
    }

    std::vector<int64_t> begin;
    std::vector<int64_t> size;
    if (param_dtype_ == kNumberTypeInt32) {
      auto begin_ptr = GetDeviceAddress<int32_t>(inputs, 1);
      auto size_ptr = GetDeviceAddress<int32_t>(inputs, kSliceInputIndex2);
      begin.assign(begin_ptr, begin_ptr + begin_shape[0]);
      size.assign(size_ptr, size_ptr + size_shape[0]);
    } else if (param_dtype_ == kNumberTypeInt64) {
      auto begin_ptr = GetDeviceAddress<int64_t>(inputs, 1);
      auto size_ptr = GetDeviceAddress<int64_t>(inputs, kSliceInputIndex2);
      begin.assign(begin_ptr, begin_ptr + begin_shape[0]);
      size.assign(size_ptr, size_ptr + size_shape[0]);
    } else {
      MS_LOG(ERROR) << "Invalid dtype: " << TypeIdLabel(param_dtype_);
      return false;
    }

    for (size_t i = 0; i < begin.size(); ++i) {
      if (input_shape[i] < begin[i] + size[i]) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', slice shape should be not greater than origin shape. But in dimension i=" << i
                      << ", origin shape 'input_shape[i]' is " << input_shape[i]
                      << " and slice shape 'LongToSize(begin[i] + size[i])' is " << LongToSize(begin[i] + size[i]);
        return false;
      }
    }
    InitSliceParam(input_shape, begin, size);
  }

  auto input_addr = inputs[kIndex0]->device_ptr();
  auto output_addr = outputs[kIndex0]->device_ptr();
  if (origin_dim_size_ == kSliceTwoDims) {
    auto task = [this, &input_addr, &output_addr](size_t start, size_t end) {
      auto src = static_cast<int8_t *>(input_addr) +
                 data_size_ * slice_param_.shape_[1] * (SizeToInt(start) + slice_param_.begin_[0]);
      auto dst = static_cast<int8_t *>(output_addr) + data_size_ * slice_param_.size_[1] * SizeToInt(start);
      SliceSimpleDim2(src, dst, &slice_param_, data_size_, end - start);
    };
    ParallelLaunchAutoSearch(task, slice_param_.size_[0], this, &parallel_search_info_);
    return true;
  }
  DoSliceNoParallel(input_addr, output_addr, &slice_param_, data_size_);

  return true;
}

#define SLICE_CPU_REGISTER_KERNEL_ATTR(DT)                                                                         \
  KernelAttr().AddInputAttr(DT).AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(DT),   \
    KernelAttr().AddInputAttr(DT).AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(DT), \
    KernelAttr()                                                                                                   \
      .AddInputAttr(DT)                                                                                            \
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)                                                            \
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)                                                            \
      .AddOutputAttr(DT),                                                                                          \
    KernelAttr()                                                                                                   \
      .AddInputAttr(DT)                                                                                            \
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)                                                            \
      .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)                                                            \
      .AddOutputAttr(DT)

std::vector<KernelAttr> SliceCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeBool),      SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeInt8),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeInt16),     SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeInt32),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeInt64),     SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeUInt8),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeUInt16),    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeUInt32),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeUInt64),    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeFloat16),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeFloat32),   SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeFloat64),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeComplex64), SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeComplex128),
    SLICE_CPU_REGISTER_KERNEL_ATTR(kNumberTypeBFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Slice, SliceCpuKernelMod);
}  // namespace slice_cpu
}  // namespace kernel
}  // namespace mindspore
