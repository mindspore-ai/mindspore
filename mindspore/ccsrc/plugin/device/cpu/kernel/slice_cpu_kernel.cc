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

#include "plugin/device/cpu/kernel/slice_cpu_kernel.h"
#include <complex>
#include <algorithm>
#include <unordered_map>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/slice.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSliceInputsNum = 1;
constexpr size_t kSliceDynamicInputNum = 3;
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

bool SliceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
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
                                                                {kNumberTypeComplex128, sizeof(std::complex<double>)}};
  TypeId dtype = inputs[0]->GetDtype();
  auto size_pair = type_size_map.find(dtype);
  if (size_pair == type_size_map.end()) {
    MS_LOG(EXCEPTION) << "Slice supports type in type_size_map, but got " << TypeIdToType(dtype)->ToString();
  }
  data_size_ = size_pair->second;
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

int SliceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Slice>(base_operator);
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  if (input_shape.size() > DIMENSION_8D || input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input tensor must be in range [1D, 8D], but got " << input_shape.size()
                      << "D.";
  }

  size_t input_num = inputs.size();
  // begin and size are const input
  if (input_num == 1) {
    auto size = kernel_ptr->get_size();
    auto begin = kernel_ptr->get_begin();
    if (begin.size() != input_shape.size() || size.size() != input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the lengths of 'begin' and 'size' must be equal to "
                           "the dimension of input tensor, but got the length of 'begin' "
                        << begin.size() << ", the length of 'size' " << size.size()
                        << "and the dimension of input tensor " << input_shape.size();
    }
    InitSliceParam(input_shape, begin, size);
  } else if (input_num == kSliceDynamicInputNum) {
    input_shape_ = inputs[0]->GetShapeVector();
    begin_shape_ = inputs[1]->GetShapeVector();
    size_shape_ = inputs[kSliceInputIndex2]->GetShapeVector();
  }
  return KRET_OK;
}

void SliceSimpleDim2(const int8_t *input, int8_t *output, const SliceParameter *param, int data_size, size_t row_size) {
  size_t copy_size = IntToSize(data_size * param->size_[1]);
  for (size_t i = 0; i < row_size; ++i) {
    auto dst = output + data_size * param->size_[1] * i;
    auto src = input + data_size * (param->shape_[1] * i + param->begin_[1]);
    auto ret = memcpy_s(dst, copy_size, src, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kKernelName << "', memcpy failed. Error no: " << ret;
    }
  }
}

bool SliceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kSliceInputsNum && inputs.size() != kSliceDynamicInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kSliceInputsNum << " or "
                      << kSliceDynamicInputNum << ", but got " << inputs.size() << " input(s).";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSliceOutputsNum, kernel_name_);

  auto input_addr = inputs[0]->addr;
  auto output_addr = outputs[0]->addr;
  if (inputs.size() == kSliceDynamicInputNum) {
    auto input_shape = input_shape_;
    auto begin_shape = begin_shape_;
    auto size_shape = size_shape_;
    if (begin_shape.size() != 1 || size_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin' and 'size' must be 1, but got the dimension of 'begin': "
                        << begin_shape.size() << " and the dimension of 'size': " << size_shape.size();
    }
    if (begin_shape[0] != SizeToLong(input_shape.size()) || size_shape[0] != SizeToLong(input_shape.size())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the lengths of 'begin' and 'size' must be equal to "
                           "the dimension of input tensor, but got the length of 'begin' "
                        << begin_shape[0] << ", the length of 'size' " << size_shape[0]
                        << "and the dimension of input tensor " << input_shape.size();
    }
    auto begin_ptr = reinterpret_cast<int32_t *>(inputs[1]->addr);
    auto size_ptr = reinterpret_cast<int32_t *>(inputs[kSliceInputIndex2]->addr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape[0]};
    std::vector<int64_t> size{size_ptr, size_ptr + size_shape[0]};
    for (size_t i = 0; i < begin.size(); ++i) {
      if (input_shape[i] < begin[i] + size[i]) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', slice shape should be not greater than origin shape. But in dimension i=" << i
                          << ", origin shape 'input_shape[i]' is " << input_shape[i]
                          << " and slice shape 'LongToSize(begin[i] + size[i])' is " << LongToSize(begin[i] + size[i]);
      }
    }
    InitSliceParam(input_shape, begin, size);
  }

  if (origin_dim_size_ == kSliceTwoDims) {
    auto task = [this, &input_addr, &output_addr](size_t start, size_t end) {
      auto src =
        static_cast<int8_t *>(input_addr) + data_size_ * slice_param_.shape_[1] * (start + slice_param_.begin_[0]);
      auto dst = static_cast<int8_t *>(output_addr) + data_size_ * slice_param_.size_[1] * start;
      SliceSimpleDim2(src, dst, &slice_param_, data_size_, end - start);
    };
    ParallelLaunchAutoSearch(task, slice_param_.size_[0], this, &parallel_search_info_);
    return true;
  }
  DoSliceNoParallel(input_addr, output_addr, &slice_param_, data_size_);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Slice, SliceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
