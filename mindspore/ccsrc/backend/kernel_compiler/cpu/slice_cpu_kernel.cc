/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
int NormalizeBeginPos(int begin_pos, int dim_len) {
  if (begin_pos < 0) {
    int normal_pos = begin_pos + dim_len;
    return std::max(normal_pos, 0);
  }
  return std::min(begin_pos, dim_len - 1);
}

void SliceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
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

bool SliceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                            const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(ERROR) << "Slice requires 1 input and 1 output, but got " << inputs.size() << " input and " << outputs.size()
                  << " output.";
    return false;
  }
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "Slice output memory size should be greater than 0, but got 0.";
    return true;
  }
  auto input_addr = inputs[0]->addr;
  auto output_addr = outputs[0]->addr;
  DoSliceNoParallel(input_addr, output_addr, &slice_param_, data_size_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
