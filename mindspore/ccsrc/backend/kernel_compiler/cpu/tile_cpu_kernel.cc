/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/tile_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void TileCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  y_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  std::vector<int64_t> multiples_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "multiples");
  (void)std::transform(multiples_me.begin(), multiples_me.end(), std::back_inserter(multiples_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  dtype_ = AnfAlgo ::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  if (dtype_ == kTypeUnknown) {
    dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  }

  launch_map_[kNumberTypeInt8] = &TileCPUKernel::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TileCPUKernel::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TileCPUKernel::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &TileCPUKernel::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TileCPUKernel::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TileCPUKernel::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TileCPUKernel::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TileCPUKernel::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat32] = &TileCPUKernel::LaunchKernel<float>;
  launch_map_[kNumberTypeBool] = &TileCPUKernel::LaunchKernel<bool>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Input data type: " << dtype_ << "is not supported for Tile kernel on CPU.";
  }
}

bool TileCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TileRecTask(const T *x, T *y, size_t dim, size_t *offset, std::vector<size_t> *pos,
                 const std::vector<int> &multiples, const std::vector<size_t> &cargo_x,
                 const std::vector<size_t> &cargo_y, const std::vector<size_t> &x_shape) {
  if (dim == x_shape.size()) {
    return;
  }
  for (size_t i = 0; i < x_shape[dim]; ++i) {
    (*pos)[dim] = i;
    if (dim == x_shape.size() - 1) {
      size_t x_offset = 0;
      for (size_t j = 0; j < (*pos).size(); ++j) {
        x_offset += (*pos)[j] * cargo_x[j];
      }
      memcpy_s(y + *offset, sizeof(T), x + x_offset, sizeof(T));
      *offset += 1;
      continue;
    }
    TileRecTask(x, y, dim + 1, offset, pos, multiples, cargo_x, cargo_y, x_shape);
  }
  size_t dim_size = cargo_y[dim] * sizeof(T);
  for (int m = 0; m < multiples[dim] - 1; ++m) {
    size_t y_offset = *offset - cargo_y[dim];
    memcpy_s(y + *offset, dim_size, y + y_offset, dim_size);
    *offset += cargo_y[dim];
  }
}

template <typename T>
void TileCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t ones = multiples_.size() - x_shape_.size();
  if (ones > 0) {
    for (size_t i = 0; i < ones; ++i) {
      x_shape_.insert(x_shape_.begin(), 1);
    }
  }
  int d = multiples_.size();
  std::vector<size_t> pos(d, 0);
  std::vector<size_t> cargo_x(d, 1);
  std::vector<size_t> cargo_y = x_shape_;
  for (int i = d - 2; i >= 0; --i) {
    cargo_x[i] = x_shape_[i + 1] * cargo_x[i + 1];
    cargo_y[i] *= cargo_y[i + 1] * multiples_[i + 1];
  }
  size_t offset = 0;
  TileRecTask<T>(x_addr, y_addr, 0, &offset, &pos, multiples_, cargo_x, cargo_y, x_shape_);
}

void TileCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but TileCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but TileCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
