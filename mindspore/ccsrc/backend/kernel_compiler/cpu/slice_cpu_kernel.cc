/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/slice_cpu_kernel.h"
#include <algorithm>
#include <unordered_map>
#include "common/thread_pool.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSliceInputsNum = 1;
constexpr size_t kSliceOutputsNum = 1;
}  // namespace

int NormalizeBeginPos(int begin_pos, int dim_len) {
  if (begin_pos < 0) {
    int normal_pos = begin_pos + dim_len;
    return std::max(normal_pos, 0);
  }
  return std::min(begin_pos, dim_len - 1);
}

void SliceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  static const std::unordered_map<TypeId, int> type_size_map = {{kNumberTypeBool, sizeof(bool)},
                                                                {kNumberTypeInt32, sizeof(int)},
                                                                {kNumberTypeFloat32, sizeof(float)},
                                                                {kNumberTypeFloat64, sizeof(double)}};
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > DIMENSION_8D || input_shape.empty()) {
    MS_LOG(EXCEPTION) << "Slice only support 1D to 8D input tensor, but got " << input_shape.size() << "D.";
  }
  auto size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
  auto begin = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  if (begin.size() != input_shape.size() || size.size() != input_shape.size()) {
    MS_LOG(EXCEPTION) << "Slice requires the length of begin and size must be equal to input dimension.";
  }
  InitSliceParam(input_shape, begin, size);

  TypeId dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto size_pair = type_size_map.find(dtype);
  if (size_pair == type_size_map.end()) {
    MS_LOG(EXCEPTION) << "Slice supports bool, int32, float32 and float64 input tensor, but got "
                      << TypeIdToType(dtype)->ToString();
  }
  data_size_ = size_pair->second;
}

void SliceCPUKernel::InitSliceParam(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                                    const std::vector<int64_t> &size) {
  origin_dim_size_ = input_shape.size();
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i < input_shape.size()) {
      int dim_len = SizeToInt(input_shape[i]);
      int begin_pos = LongToInt(begin[i]);
      int slice_size = LongToInt(size[i]);
      if (slice_size <= 0) {
        MS_LOG(EXCEPTION) << "Slice requires the each dimension slice size must be greater than 0.";
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
      MS_LOG(EXCEPTION) << "Memcpy failed.";
    }
  }
}

bool SliceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSliceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSliceOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "Slice output memory size should be greater than 0, but got 0.";
    return true;
  }

  auto input_addr = inputs[0]->addr;
  auto output_addr = outputs[0]->addr;
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
