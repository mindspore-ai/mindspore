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
#include "plugin/device/cpu/kernel/grid_sampler_3d_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
const size_t kDataSizeThreshold = 64 * 1024;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t kFour = 4;
const size_t kFive = 5;
const size_t kInputsNum = 2;
const size_t kOutputsNum = 1;
}  // namespace

namespace mindspore {
namespace kernel {
bool GridSampler3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  dtype_ = inputs[kZero]->GetDtype();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::GridSampler3D>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  interpolation_mode_ = kernel_ptr->get_interpolation_mode();
  padding_mode_ = kernel_ptr->get_padding_mode();
  align_corners_ = kernel_ptr->get_align_corners();
  return true;
}

int GridSampler3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[kZero]->GetDeviceShapeAdaptively();
  grid_shape_ = inputs[kOne]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kZero]->GetDeviceShapeAdaptively();
  x_stride_.clear();
  grid_stride_.clear();
  output_stride_.clear();

  output_number_ = static_cast<size_t>(output_shape_[kZero] * output_shape_[kOne] * output_shape_[kTwo] *
                                       output_shape_[kThree] * output_shape_[kFour]);
  size_t stride_tmp = kOne;
  auto stride_compute = [&](std::vector<size_t> &stride, std::vector<int64_t> shape) {
    for (int i = kFour; i > -static_cast<int>(kOne); i--) {
      (void)stride.insert(stride.begin(), stride_tmp);
      stride_tmp *= LongToSize(shape[IntToSize(i)]);
    }
    stride_tmp = kOne;
  };
  stride_compute(x_stride_, x_shape_);
  stride_compute(grid_stride_, grid_shape_);
  stride_compute(output_stride_, output_shape_);
  return ret;
}

bool GridSampler3DCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Input dtype only support float32 and float64!";
  }
  return true;
}

template <typename T>
void GridSampler3DCpuKernelMod::ComputeTask(T *x_addr, T *grid_addr, T *output_addr, const size_t &seq) {
  size_t out_iter[kFive] = {kZero, seq, kZero, kZero, kZero};
  size_t count = kFour;
  while (out_iter[kOne] > kZero) {
    if (count == kOne) {
      count--;
    }
    out_iter[count] = LongToUlong(out_iter[kOne] % output_shape_[count]);
    out_iter[kOne] /= LongToSize(output_shape_[count--]);
  }
  const size_t out_c = LongToSize(output_shape_[kOne]);
  int64_t grid_offset =
    static_cast<int64_t>(out_iter[kZero] * grid_stride_[kZero] + out_iter[kTwo] * grid_stride_[kOne] +
                         out_iter[kThree] * grid_stride_[kTwo] + out_iter[kFour] * grid_stride_[kThree]);
  T x = grid_addr[grid_offset];
  T y = grid_addr[static_cast<size_t>(grid_offset) + grid_stride_[kFour]];
  T z = grid_addr[static_cast<size_t>(grid_offset + kTwo * grid_stride_[kFour])];
  x = grid_sampler_compute_source_index(x, x_shape_[kFour], padding_mode_, align_corners_);
  y = grid_sampler_compute_source_index(y, x_shape_[kThree], padding_mode_, align_corners_);
  z = grid_sampler_compute_source_index(z, x_shape_[kTwo], padding_mode_, align_corners_);
  auto x_ptr_NC = out_iter[kZero] * x_stride_[kZero];
  auto output_ptr_NCDHW = out_iter[kZero] * output_stride_[kZero] + out_iter[kTwo] * output_stride_[kTwo] +
                          out_iter[kThree] * output_stride_[kThree] + out_iter[kFour] * output_stride_[kFour];
  if (interpolation_mode_ == "bilinear") {
    int64_t x_tnw = static_cast<int64_t>(std::floor(x));
    int64_t y_tnw = static_cast<int64_t>(std::floor(y));
    int64_t z_tnw = static_cast<int64_t>(std::floor(z));
    int64_t x_tne = static_cast<int64_t>(x_tnw + kOne), y_tne = y_tnw, z_tne = z_tnw;
    int64_t x_tsw = x_tnw, y_tsw = static_cast<int64_t>(y_tnw + kOne), z_tsw = z_tnw;
    int64_t x_tse = static_cast<int64_t>(x_tnw + kOne), y_tse = static_cast<int64_t>(y_tnw + kOne), z_tse = z_tnw;
    int64_t x_bnw = x_tnw, y_bnw = y_tnw, z_bnw = static_cast<int64_t>(z_tnw + kOne);
    int64_t x_bne = static_cast<int64_t>(x_tnw + kOne), y_bne = y_tnw, z_bne = static_cast<int64_t>(z_tnw + kOne);
    int64_t x_bsw = x_tnw, y_bsw = static_cast<int64_t>(y_tnw + kOne), z_bsw = static_cast<int64_t>(z_tnw + kOne);
    int64_t x_bse = static_cast<int64_t>(x_tnw + kOne), y_bse = static_cast<int64_t>(y_tnw + kOne);
    int64_t z_bse = static_cast<int64_t>(z_tnw + kOne);
    T tnw = (x_bse - x) * (y_bse - y) * (z_bse - z), tne = (x - x_bsw) * (y_bsw - y) * (z_bsw - z);
    T tsw = (x_bne - x) * (y - y_bne) * (z_bne - z), tse = (x - x_bnw) * (y - y_bnw) * (z_bnw - z);
    T bnw = (x_tse - x) * (y_tse - y) * (z - z_tse), bne = (x - x_tsw) * (y_tsw - y) * (z - z_tsw);
    T bsw = (x_tne - x) * (y - y_tne) * (z - z_tne), bse = (x - x_tnw) * (y - y_tnw) * (z - z_tnw);
    for (size_t c = kZero; c < out_c; c++, x_ptr_NC += x_stride_[kOne], output_ptr_NCDHW += output_stride_[kOne]) {
      output_addr[output_ptr_NCDHW] = static_cast<T>(kZero);
      if (within_bounds_3d(z_tnw, y_tnw, x_tnw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_tnw * x_stride_[kTwo] + y_tnw * x_stride_[kThree] +
                                           x_tnw * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tnw;
      }
      if (within_bounds_3d(z_tne, y_tne, x_tne, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_tne * x_stride_[kTwo] + y_tne * x_stride_[kThree] +
                                           x_tne * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tne;
      }
      if (within_bounds_3d(z_tsw, y_tsw, x_tsw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_tsw * x_stride_[kTwo] + y_tsw * x_stride_[kThree] +
                                           x_tsw * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tsw;
      }
      if (within_bounds_3d(z_tse, y_tse, x_tse, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_tse * x_stride_[kTwo] + y_tse * x_stride_[kThree] +
                                           x_tse * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tse;
      }
      if (within_bounds_3d(z_bnw, y_bnw, x_bnw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_bnw * x_stride_[kTwo] + y_bnw * x_stride_[kThree] +
                                           x_bnw * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * bnw;
      }
      if (within_bounds_3d(z_bne, y_bne, x_bne, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_bne * x_stride_[kTwo] + y_bne * x_stride_[kThree] +
                                           x_bne * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * bne;
      }
      if (within_bounds_3d(z_bsw, y_bsw, x_bsw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_bsw * x_stride_[kTwo] + y_bsw * x_stride_[kThree] +
                                           x_bsw * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * bsw;
      }
      if (within_bounds_3d(z_bse, y_bse, x_bse, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_bse * x_stride_[kTwo] + y_bse * x_stride_[kThree] +
                                           x_bse * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * bse;
      }
    }
  } else if (interpolation_mode_ == "nearest") {
    int64_t x_nearest = static_cast<int64_t>(std::round(x));
    int64_t y_nearest = static_cast<int64_t>(std::round(y));
    int64_t z_nearest = static_cast<int64_t>(std::round(z));
    for (size_t c = kZero; c < out_c; c++, x_ptr_NC += x_stride_[kOne], output_ptr_NCDHW += output_stride_[kOne]) {
      if (within_bounds_3d(z_nearest, y_nearest, x_nearest, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
        auto x_index = static_cast<size_t>(x_ptr_NC + z_nearest * x_stride_[kTwo] + y_nearest * x_stride_[kThree] +
                                           x_nearest * x_stride_[kFour]);
        output_addr[output_ptr_NCDHW] = x_addr[x_index];
      } else {
        output_addr[output_ptr_NCDHW] = static_cast<T>(kZero);
      }
    }
  }
}

template <typename T>
void GridSampler3DCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto x_data_addr = static_cast<T *>(inputs[kZero]->addr);
  auto grid_data_addr = static_cast<T *>(inputs[kOne]->addr);
  auto output_data_addr = static_cast<T *>(outputs[kZero]->addr);
  size_t loop_count =
    LongToSize(output_shape_[kZero] * output_shape_[kTwo] * output_shape_[kThree] * output_shape_[kFour]);
  auto task = [this, &x_data_addr, &grid_data_addr, &output_data_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      ComputeTask<T>(x_data_addr, grid_data_addr, output_data_addr, i);
    }
  };
  if (output_number_ < kDataSizeThreshold) {
    task(kZero, loop_count);
  } else {
    CPUKernelUtils::ParallelFor(task, loop_count);
  }
}

template <typename T>
T GridSampler3DCpuKernelMod::grid_sampler_compute_source_index(T coord, int64_t size, const std::string &padding_mode,
                                                               bool align_corners) {
  if (align_corners) {
    coord = ((coord + 1.f) / kTwo) * (static_cast<size_t>(size) - kOne);
  } else {
    coord = ((coord + 1.f) * size - kOne) / kTwo;
  }
  if (padding_mode == "border") {
    coord = std::min(static_cast<T>(static_cast<size_t>(size) - kOne), std::max(coord, static_cast<T>(kZero)));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates(coord, static_cast<int64_t>(kZero), kTwo * (size - kOne));
    } else {
      coord = reflect_coordinates(coord, -static_cast<int64_t>(kOne), kTwo * size - kOne);
    }
    coord = std::min(static_cast<T>(static_cast<size_t>(size) - kOne), std::max(coord, static_cast<T>(kZero)));
  }
  return coord;
}

template <typename T>
T GridSampler3DCpuKernelMod::reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high) const {
  if (twice_low == twice_high) {
    return static_cast<T>(kZero);
  }
  T min = static_cast<T>(twice_low) / kTwo;
  T span = static_cast<T>(twice_high - twice_low) / kTwo;
  coord = std::fabs(coord - min);
  T extra = std::fmod(coord, span);
  int64_t flips = static_cast<int64_t>(std::floor(coord / span));
  if (static_cast<size_t>(flips) % kTwo == kZero) {
    return extra + min;
  } else {
    return (span - extra) + min;
  }
}

bool GridSampler3DCpuKernelMod::within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H,
                                                 int64_t W) const {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GridSampler3D, GridSampler3DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
