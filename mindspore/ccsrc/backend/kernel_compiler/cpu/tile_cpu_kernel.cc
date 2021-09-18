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
namespace {
constexpr size_t kTileInputsNum = 1;
constexpr size_t kTileOutputsNum = 1;
}  // namespace

void TileCPUKernel::TileMultipleCompute() {
  int large_one_multiple_count_ = 0;
  int multiple = 0;
  size_t mul_index = 0;
  for (size_t i = 0; i < multiples_.size(); i++) {
    tile_parameter_.multiples_[i] = multiples_[i];
    if (tile_parameter_.multiples_[i] > 1) {
      large_one_multiple_count_++;
      multiple = tile_parameter_.multiples_[i];
      mul_index = i;
    }
  }

  one_dim_tile_ = large_one_multiple_count_ == 1;
  if (one_dim_tile_) {
    tile_parameter_.fast_multiple_ = static_cast<size_t>(multiple);
    tile_parameter_.fast_stride_ = static_cast<size_t>(x_shape_[mul_index] * tile_parameter_.in_strides_[mul_index]);
    if (tile_parameter_.fast_stride_ == 0) {
      MS_LOG(EXCEPTION) << "Fast stride is equal to 0";
    }
    tile_parameter_.fast_outer_size_ = static_cast<size_t>(input_size_ / tile_parameter_.fast_stride_);
  }
}

void TileCPUKernel::TileTensorParamrInit(const CNodePtr &kernel_node) {
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  y_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (x_shape_.size() > MAX_TILE_DIM_SIZE || x_shape_.size() > y_shape_.size()) {
    MS_LOG(EXCEPTION) << "Tile input shape should not be greater than default max size :" << MAX_TILE_DIM_SIZE
                      << " and output shape : " << y_shape_.size() << ", but got input shape " << x_shape_.size();
  }
  std::vector<int64_t> multiples_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "multiples");
  (void)std::transform(multiples_me.begin(), multiples_me.end(), std::back_inserter(multiples_),
                       [](const int64_t &value) { return LongToInt(value); });
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t ones = multiples_.size() - x_shape_.size();
  if (ones > 0) {
    for (size_t i = 0; i < ones; ++i) {
      (void)x_shape_.insert(x_shape_.begin(), 1);
    }
  }

  input_size_ = 1;
  tile_parameter_.in_dim_ = x_shape_.size();
  for (int i = 0; i < tile_parameter_.in_dim_; i++) {
    input_size_ *= x_shape_[i];
    tile_parameter_.in_shape_[i] = x_shape_[i];
    tile_parameter_.out_shape_[i] = y_shape_[i];
  }

  int stridex = 1;
  int stridey = 1;
  for (int i = tile_parameter_.in_dim_ - 1; i >= 0; i--) {
    tile_parameter_.in_strides_[i] = stridex;
    tile_parameter_.out_strides_[i] = stridey;
    stridex *= x_shape_[i];
    stridey *= y_shape_[i];
  }

  TileMultipleCompute();
}

void TileCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  TileTensorParamrInit(kernel_node);

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
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
}

bool TileCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTileInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTileOutputsNum, kernel_name_);
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TileCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  tile_parameter_.data_size_ = sizeof(T);

  if (one_dim_tile_) {
    auto task = [&x_addr, &y_addr, this](size_t start, size_t end) {
      TileSimple(x_addr, y_addr, start, end, &tile_parameter_);
    };
    ParallelLaunchAutoSearch(task, tile_parameter_.fast_outer_size_, this, &parallel_search_info_);
    return;
  }

  Tile(x_addr, y_addr, &tile_parameter_);
}
}  // namespace kernel
}  // namespace mindspore
