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

#include "plugin/device/cpu/kernel/tile_cpu_kernel.h"
#include <algorithm>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTileInputsNum = 1;
constexpr size_t kTileDynamicInputsNum = 2;
constexpr size_t kTileOutputsNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
}  // namespace

void TileCpuKernelMod::TileMultipleCompute() {
  size_t ones = multiples_.size() - x_shape_.size();
  if (ones > 0) {
    for (size_t i = 0; i < ones; ++i) {
      x_shape_.insert(x_shape_.begin(), 1);
    }
  }
  if (x_shape_.size() > MAX_TILE_DIM_SIZE || x_shape_.size() > y_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', input shape can not be greater than default max size: " << MAX_TILE_DIM_SIZE
                      << " and output shape: " << y_shape_.size() << ", but got input shape " << x_shape_.size();
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
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fast stride can not be equal to 0";
    }
    tile_parameter_.fast_outer_size_ = static_cast<size_t>(input_size_ / tile_parameter_.fast_stride_);
  }
}

bool TileCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();
  input_num = inputs.size();
  dtype_ = inputs[kIndex0]->GetDtype();
  multiples_.clear();
  if (input_num == kTileInputsNum) {
    std::vector<int64_t> multiples_me = GetValue<std::vector<int64_t>>(prim->GetAttr("multiples"));
    (void)std::transform(multiples_me.begin(), multiples_me.end(), std::back_inserter(multiples_),
                         [](const int64_t &value) { return static_cast<int>(value); });
  }

  launch_map_[kNumberTypeInt8] = &TileCpuKernelMod::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TileCpuKernelMod::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TileCpuKernelMod::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &TileCpuKernelMod::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TileCpuKernelMod::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TileCpuKernelMod::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TileCpuKernelMod::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TileCpuKernelMod::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat32] = &TileCpuKernelMod::LaunchKernel<float>;
  launch_map_[kNumberTypeFloat16] = &TileCpuKernelMod::LaunchKernel<float16>;
  launch_map_[kNumberTypeFloat64] = &TileCpuKernelMod::LaunchKernel<double>;
  launch_map_[kNumberTypeComplex64] = &TileCpuKernelMod::LaunchKernel<complex64>;
  launch_map_[kNumberTypeComplex128] = &TileCpuKernelMod::LaunchKernel<complex128>;
  launch_map_[kNumberTypeBool] = &TileCpuKernelMod::LaunchKernel<bool>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input must be bool, int, float, uint or complex, but got "
                      << TypeIdToType(dtype_)->ToString();
    return false;
  }
  return true;
}

int TileCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[kIndex0]->GetShapeVector();
  y_shape_ = outputs[kIndex0]->GetShapeVector();
  if (input_num == kTileDynamicInputsNum) {
    multiple_shape = inputs[kIndex1]->GetShapeVector();
    multiple_dtype_ = inputs[kIndex1]->GetDtype();
  }
  return KRET_OK;
}

bool TileCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kTileInputsNum && inputs.size() != kTileDynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of input must be " << kTileInputsNum << " or "
                      << kTileDynamicInputsNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTileOutputsNum, kernel_name_);
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TileCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (input_num == kTileInputsNum) {
    TileMultipleCompute();
  }
  if (input_num == kTileDynamicInputsNum) {
    multiples_.clear();
    int64_t multiple_nums = 1;
    for (size_t i = 0; i < multiple_shape.size(); ++i) {
      multiple_nums *= multiple_shape[i];
    }
    if (multiple_dtype_ == kNumberTypeInt32) {
      auto multiples_addr = GetDeviceAddress<int32_t>(inputs, 1);
      for (size_t i = 0; i < LongToSize(multiple_nums); ++i) {
        (void)multiples_.emplace_back(multiples_addr[i]);
      }
    } else {
      auto multiples_addr = GetDeviceAddress<int64_t>(inputs, 1);
      for (size_t i = 0; i < LongToSize(multiple_nums); ++i) {
        (void)multiples_.emplace_back(multiples_addr[i]);
      }
    }
    TileMultipleCompute();
  }

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

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Tile, TileCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
