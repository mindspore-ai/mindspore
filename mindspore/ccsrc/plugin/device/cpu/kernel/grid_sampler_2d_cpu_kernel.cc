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

#include "plugin/device/cpu/kernel/grid_sampler_2d_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
const size_t kDataSizeThreshold = 64 * 1024;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
}  // namespace

namespace mindspore {
namespace kernel {
void GridSampler2DCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  grid_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output_number_ = output_shape_[kZero] * output_shape_[kOne] * output_shape_[kTwo] * output_shape_[kThree];
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t stride_tmp = 1;
  auto stride_compute = [this, &stride_tmp](std::vector<size_t> &stride, std::vector<size_t> shape) {
    for (int32_t i = 3; i > -1; i--) {
      stride.insert(stride.begin(), stride_tmp);
      stride_tmp *= shape[i];
    }
    stride_tmp = 1;
  };
  stride_compute(x_stride_, x_shape_);
  stride_compute(grid_stride_, grid_shape_);
  stride_compute(output_stride_, output_shape_);
  interpolation_mode_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "interpolation_mode");
  padding_mode_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "padding_mode");
  align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
}

bool GridSampler2DCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Input dtype only support float16, float32!, but got" << dtype_;
  }
  return true;
}

template <typename T>
void GridSampler2DCpuKernelMod::ComputeTask(T *x_addr, T *grid_addr, T *output_addr, const size_t &seq) {
  size_t out_iter[4] = {0, seq, 0, 0};
  size_t count = 3;
  while (out_iter[1] > 0) {
    if (count == 1) {
      count--;
    }
    out_iter[count] = out_iter[kOne] % output_shape_[count];
    out_iter[1] /= output_shape_[count--];
  }
  const size_t out_c = output_shape_[kOne];
  int64_t grid_offset =
    out_iter[kZero] * grid_stride_[kZero] + out_iter[kTwo] * grid_stride_[kOne] + out_iter[kThree] * grid_stride_[kTwo];
  T x = grid_addr[grid_offset];
  T y = grid_addr[grid_offset + grid_stride_[kThree]];
  x = GridSamplerComputerSourceIndex(x, x_shape_[kThree], padding_mode_, align_corners_);
  y = GridSamplerComputerSourceIndex(y, x_shape_[kTwo], padding_mode_, align_corners_);
  auto x_ptr_NC = out_iter[kZero] * x_stride_[kZero];
  auto output_ptr_NCDHW = out_iter[kZero] * output_stride_[kZero] + out_iter[kTwo] * output_stride_[kTwo] +
                          out_iter[kThree] * output_stride_[kThree];
  if (interpolation_mode_ == "bilinear") {
    int64_t x_tnw = static_cast<int64_t>(std::floor(x));
    int64_t y_tnw = static_cast<int64_t>(std::floor(y));
    int64_t x_tne = x_tnw + 1;
    int64_t y_tne = y_tnw;
    int64_t x_tsw = x_tnw;
    int64_t y_tsw = y_tnw + 1;
    int64_t x_tse = x_tnw + 1;
    int64_t y_tse = y_tnw + 1;

    T tnw = (x_tse - x) * (y_tse - y);
    T tne = (x - x_tsw) * (y_tsw - y);
    T tsw = (x_tne - x) * (y - y_tne);
    T tse = (x - x_tnw) * (y - y_tnw);

    for (size_t c = 0; c < out_c; c++, x_ptr_NC += x_stride_[1], output_ptr_NCDHW += output_stride_[1]) {
      output_addr[output_ptr_NCDHW] = static_cast<T>(0);
      if (WithinBounds2D(y_tnw, x_tnw, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tnw * x_stride_[kTwo] + x_tnw * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tnw;
      }
      if (WithinBounds2D(y_tne, x_tne, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tne * x_stride_[kTwo] + x_tne * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tne;
      }
      if (WithinBounds2D(y_tsw, x_tsw, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tsw * x_stride_[kTwo] + x_tsw * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tsw;
      }
      if (WithinBounds2D(y_tse, x_tse, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tse * x_stride_[kTwo] + x_tse * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tse;
      }
    }
  } else if (interpolation_mode_ == "nearest") {
    int64_t x_nearest = static_cast<int64_t>(std::round(x));
    int64_t y_nearest = static_cast<int64_t>(std::round(y));
    for (size_t c = 0; c < out_c; c++, x_ptr_NC += x_stride_[kOne], output_ptr_NCDHW += output_stride_[kOne]) {
      if (WithinBounds2D(y_nearest, x_nearest, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_nearest * x_stride_[kTwo] + x_nearest * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] = x_addr[x_index];
      } else {
        output_addr[output_ptr_NCDHW] = static_cast<T>(0);
      }
    }
  }
}

void GridSampler2DCpuKernelMod::ComputeTask(float16 *x_addr, float16 *grid_addr, float16 *output_addr,
                                            const size_t &seq) {
  size_t out_iter[4] = {0, seq, 0, 0};
  size_t count = 3;
  while (out_iter[1] > 0) {
    if (count == 1) {
      count--;
    }
    out_iter[count] = out_iter[kOne] % output_shape_[count];
    out_iter[1] /= output_shape_[count--];
  }
  const size_t out_c = output_shape_[kOne];
  int64_t grid_offset =
    out_iter[kZero] * grid_stride_[kZero] + out_iter[kTwo] * grid_stride_[kOne] + out_iter[kThree] * grid_stride_[kTwo];
  float x = static_cast<float>(grid_addr[grid_offset]);
  float y = static_cast<float>(grid_addr[grid_offset + grid_stride_[kThree]]);
  x = GridSamplerComputerSourceIndex(x, x_shape_[kThree], padding_mode_, align_corners_);
  y = GridSamplerComputerSourceIndex(y, x_shape_[kTwo], padding_mode_, align_corners_);
  auto x_ptr_NC = out_iter[kZero] * x_stride_[kZero];
  auto output_ptr_NCDHW = out_iter[0] * output_stride_[kZero] + out_iter[kTwo] * output_stride_[kTwo] +
                          out_iter[kThree] * output_stride_[kThree];
  if (interpolation_mode_ == "bilinear") {
    int64_t x_tnw = static_cast<int64_t>(std::floor(x));
    int64_t y_tnw = static_cast<int64_t>(std::floor(y));
    int64_t x_tne = x_tnw + 1;
    int64_t y_tne = y_tnw;
    int64_t x_tsw = x_tnw;
    int64_t y_tsw = y_tnw + 1;
    int64_t x_tse = x_tnw + 1;
    int64_t y_tse = y_tnw + 1;

    float16 tnw = static_cast<float16>((x_tse - x) * (y_tse - y));
    float16 tne = static_cast<float16>((x - x_tsw) * (y_tsw - y));
    float16 tsw = static_cast<float16>((x_tne - x) * (y - y_tne));
    float16 tse = static_cast<float16>((x - x_tnw) * (y - y_tnw));

    for (size_t c = 0; c < out_c; c++, x_ptr_NC += x_stride_[1], output_ptr_NCDHW += output_stride_[1]) {
      output_addr[output_ptr_NCDHW] = static_cast<float16>(0);
      if (WithinBounds2D(y_tnw, x_tnw, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tnw * x_stride_[2] + x_tnw * x_stride_[3];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tnw;
      }
      if (WithinBounds2D(y_tne, x_tne, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tne * x_stride_[2] + x_tne * x_stride_[3];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tne;
      }
      if (WithinBounds2D(y_tsw, x_tsw, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tsw * x_stride_[2] + x_tsw * x_stride_[3];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tsw;
      }
      if (WithinBounds2D(y_tse, x_tse, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_tse * x_stride_[2] + x_tse * x_stride_[3];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tse;
      }
    }
  } else if (interpolation_mode_ == "nearest") {
    int64_t x_nearest = static_cast<int64_t>(std::round(x));
    int64_t y_nearest = static_cast<int64_t>(std::round(y));
    for (size_t c = 0; c < out_c; c++, x_ptr_NC += x_stride_[1], output_ptr_NCDHW += output_stride_[1]) {
      if (WithinBounds2D(y_nearest, x_nearest, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + y_nearest * x_stride_[2] + x_nearest * x_stride_[3];
        output_addr[output_ptr_NCDHW] = x_addr[x_index];
      } else {
        output_addr[output_ptr_NCDHW] = static_cast<float16>(0);
      }
    }
  }
}

template <typename T>
void GridSampler2DCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto x_data_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto grid_data_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_data_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t loop_count = output_shape_[0] * output_shape_[2] * output_shape_[3];
  auto task = [this, &x_data_addr, &grid_data_addr, &output_data_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      ComputeTask<T>(x_data_addr, grid_data_addr, output_data_addr, i);
    }
  };
  if (output_number_ < kDataSizeThreshold) {
    task(0, loop_count);
  } else {
    CPUKernelUtils::ParallelFor(task, loop_count);
  }
}

void GridSampler2DCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto x_data_addr = reinterpret_cast<float16 *>(inputs[0]->addr);
  auto grid_data_addr = reinterpret_cast<float16 *>(inputs[1]->addr);
  auto output_data_addr = reinterpret_cast<float16 *>(outputs[0]->addr);
  size_t loop_count = output_shape_[0] * output_shape_[2] * output_shape_[3];
  auto task = [this, &x_data_addr, &grid_data_addr, &output_data_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      ComputeTask(x_data_addr, grid_data_addr, output_data_addr, i);
    }
  };
  if (output_number_ < kDataSizeThreshold) {
    task(0, loop_count);
  } else {
    CPUKernelUtils::ParallelFor(task, loop_count);
  }
}

template <typename T>
T GridSampler2DCpuKernelMod::GridSamplerComputerSourceIndex(T coord, int64_t size, std::string padding_mode,
                                                            bool align_corners) {
  const int64_t num2 = 2;
  if (align_corners) {
    coord = ((coord + 1.f) / num2) * (size - 1);
  } else {
    coord = ((coord + 1.f) * size - 1) / num2;
  }
  if (padding_mode == "border") {
    coord = std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = ReflectCoordinates(coord, 0, num2 * (size - 1));
    } else {
      coord = ReflectCoordinates(coord, -1, num2 * size - 1);
    }
    coord = std::min(static_cast<T>(size - 1), std::max(coord, static_cast<T>(0)));
  }
  return coord;
}

template <typename T>
T GridSampler2DCpuKernelMod::ReflectCoordinates(T coord, int64_t twice_low, int64_t twice_high) {
  const int64_t num2 = 2;
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  coord = std::fabs(coord - min);
  T extra = std::fmod(coord, span);
  int64_t flips = static_cast<int64_t>(std::floor(coord / span));
  if (flips % num2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

bool GridSampler2DCpuKernelMod::WithinBounds2D(int64_t h, int64_t w, int64_t H, int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GridSampler2D, GridSampler2DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
