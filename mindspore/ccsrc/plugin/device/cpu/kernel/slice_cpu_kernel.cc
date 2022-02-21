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
#include <algorithm>
#include <unordered_map>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSliceInputsNum = 1;
constexpr size_t kSliceDynamicInputNum = 3;
constexpr size_t kSliceOutputsNum = 1;
constexpr char kKernelName[] = "Slice";
}  // namespace

int NormalizeBeginPos(int begin_pos, int dim_len) {
  if (begin_pos < 0) {
    int normal_pos = begin_pos + dim_len;
    return std::max(normal_pos, 0);
  }
  return std::min(begin_pos, dim_len - 1);
}

void SliceCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  static const std::unordered_map<TypeId, int> type_size_map = {{kNumberTypeBool, sizeof(bool)},
                                                                {kNumberTypeInt32, sizeof(int)},
                                                                {kNumberTypeFloat32, sizeof(float)},
                                                                {kNumberTypeFloat64, sizeof(double)}};
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > DIMENSION_8D || input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input tensor should be in range [1D, 8D], but got " << input_shape.size()
                      << "D.";
  }

  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  // begin and size are const input
  if (input_num == 1) {
    auto size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
    auto begin = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
    if (begin.size() != input_shape.size() || size.size() != input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the lengths of 'begin' and 'size' should be equal to "
                           "the dimension of input tensor, but got the length of 'begin' "
                        << begin.size() << ", the length of 'size' " << size.size()
                        << "and the dimension of input tensor " << input_shape.size();
    }
    InitSliceParam(input_shape, begin, size);
  }

  TypeId dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto size_pair = type_size_map.find(dtype);
  if (size_pair == type_size_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be bool, int32, float32 or float64, but got "
                      << TypeIdToType(dtype)->ToString();
  }
  data_size_ = size_pair->second;
}

void SliceCpuKernelMod::InitSliceParam(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                                       const std::vector<int64_t> &size) {
  origin_dim_size_ = input_shape.size();
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i < input_shape.size()) {
      int dim_len = SizeToInt(input_shape[i]);
      int begin_pos = LongToInt(begin[i]);
      int slice_size = LongToInt(size[i]);
      if (slice_size == -1) {
        slice_size = dim_len - begin_pos;
      }
      if (slice_size <= 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the each dimension of 'size' should be greater than 0 "
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
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kSliceInputsNum << " or "
                      << kSliceDynamicInputNum << ", but got " << inputs.size() << " input(s).";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSliceOutputsNum, kernel_name_);

  auto input_addr = inputs[0]->addr;
  auto output_addr = outputs[0]->addr;
  if (inputs.size() == kSliceDynamicInputNum) {
    auto cnode = cnode_ptr_.lock();
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
    auto begin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 1);
    auto size_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 2);
    if (begin_shape.size() != 1 || size_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin' and 'size' should be 1, but got the dimension of 'begin': "
                        << begin_shape.size() << " and the dimension of 'size': " << size_shape.size();
    }
    if (begin_shape[0] != input_shape.size() || size_shape[0] != input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the lengths of 'begin' and 'size' should be equal to "
                           "the dimension of input tensor, but got the length of 'begin' "
                        << begin_shape[0] << ", the length of 'size' " << size_shape[0]
                        << "and the dimension of input tensor " << input_shape.size();
    }
    auto begin_ptr = reinterpret_cast<int32_t *>(inputs[1]->addr);
    auto size_ptr = reinterpret_cast<int32_t *>(inputs[2]->addr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape[0]};
    std::vector<int64_t> size{size_ptr, size_ptr + size_shape[0]};
    for (size_t i = 0; i < begin.size(); ++i) {
      if (input_shape[i] < LongToSize(begin[i] + size[i])) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', slice shape should be not greater than origin shape. But in dimension i=" << i
                          << ", origin shape 'input_shape[i]' is " << input_shape[i] << " and slice shape is "
                          << LongToSize(begin[i] + size[i]);
      }
    }
    InitSliceParam(input_shape, begin, size);
  }

  if (origin_dim_size_ == 2) {
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
}  // namespace kernel
}  // namespace mindspore
