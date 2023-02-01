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
const size_t kInputsNum = 2;
const size_t kOutputsNum = 1;
}  // namespace

namespace mindspore {
namespace kernel {
bool GridSampler2DCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  dtype_ = inputs[kZero]->GetDtype();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::GridSampler2D>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  interpolation_mode_ = kernel_ptr->get_interpolation_mode();
  padding_mode_ = kernel_ptr->get_padding_mode();
  align_corners_ = kernel_ptr->get_align_corners();
  return true;
}

int GridSampler2DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[kZero]->GetDeviceShapeAdaptively();
  grid_shape_ = inputs[kOne]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kZero]->GetDeviceShapeAdaptively();
  output_number_ = LongToSize(output_shape_[kZero] * output_shape_[kOne] * output_shape_[kTwo] * output_shape_[kThree]);
  x_stride_.clear();
  grid_stride_.clear();
  output_stride_.clear();

  size_t stride_tmp = 1;
  auto stride_compute = [this, &stride_tmp](std::vector<size_t> &stride, ShapeVector shape) {
    for (int32_t i = 3; i > -1; i--) {
      (void)stride.insert(stride.begin(), stride_tmp);
      stride_tmp *= LongToSize(shape[IntToSize(i)]);
    }
    stride_tmp = 1;
  };
  stride_compute(x_stride_, x_shape_);
  stride_compute(grid_stride_, grid_shape_);
  stride_compute(output_stride_, output_shape_);
  return ret;
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
void GridSampler2DCpuKernelMod::ComputeTask(const T *x_addr, const T *grid_addr, T *output_addr, const size_t &seq) {
  size_t out_iter[4] = {0, seq, 0, 0};
  size_t count = 3;
  while (out_iter[1] > 0) {
    if (count == 1) {
      count--;
    }
    out_iter[count] = out_iter[kOne] % LongToSize(output_shape_[count]);
    out_iter[1] /= LongToSize(output_shape_[count]);
    if (count == 0) {
      break;
    }
    count--;
  }
  const size_t out_c = LongToSize(output_shape_[kOne]);
  size_t grid_offset =
    out_iter[kZero] * grid_stride_[kZero] + out_iter[kTwo] * grid_stride_[kOne] + out_iter[kThree] * grid_stride_[kTwo];
  T x = grid_addr[grid_offset];
  T y = grid_addr[grid_offset + grid_stride_[kThree]];
  x = GridSamplerComputeSourceIndex(x, x_shape_[kThree], padding_mode_, align_corners_);
  y = GridSamplerComputeSourceIndex(y, x_shape_[kTwo], padding_mode_, align_corners_);
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
        auto x_index = x_ptr_NC + LongToSize(y_tnw) * x_stride_[kTwo] + LongToSize(x_tnw) * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tnw;
      }
      if (WithinBounds2D(y_tne, x_tne, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + LongToSize(y_tne) * x_stride_[kTwo] + LongToSize(x_tne) * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tne;
      }
      if (WithinBounds2D(y_tsw, x_tsw, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + LongToSize(y_tsw) * x_stride_[kTwo] + LongToSize(x_tsw) * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tsw;
      }
      if (WithinBounds2D(y_tse, x_tse, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + LongToSize(y_tse) * x_stride_[kTwo] + LongToSize(x_tse) * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] += x_addr[x_index] * tse;
      }
    }
  } else if (interpolation_mode_ == "nearest") {
    int64_t x_nearest = static_cast<int64_t>(std::round(x));
    int64_t y_nearest = static_cast<int64_t>(std::round(y));
    for (size_t c = 0; c < out_c; c++, x_ptr_NC += x_stride_[kOne], output_ptr_NCDHW += output_stride_[kOne]) {
      if (WithinBounds2D(y_nearest, x_nearest, x_shape_[kTwo], x_shape_[kThree])) {
        auto x_index = x_ptr_NC + LongToSize(y_nearest) * x_stride_[kTwo] + LongToSize(x_nearest) * x_stride_[kThree];
        output_addr[output_ptr_NCDHW] = x_addr[x_index];
      } else {
        output_addr[output_ptr_NCDHW] = static_cast<T>(0);
      }
    }
  }
}

void GridSampler2DCpuKernelMod::ComputeTask(const float16 *x_addr, const float16 *grid_addr, float16 *output_addr,
                                            const size_t &seq) {
  int64_t y_stride[4];
  int64_t stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    y_stride[i] = stride_tmp;
    stride_tmp *= output_shape_[i];
  }

  int64_t x_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    x_stride[i] = stride_tmp;
    stride_tmp *= x_shape_[i];
  }

  int64_t grid_stride[4];
  stride_tmp = 1;
  for (int32_t i = 3; i > -1; i--) {
    grid_stride[i] = stride_tmp;
    stride_tmp *= grid_shape_[i];
  }

  Call2Half(x_addr, output_addr, grid_addr, x_shape_, output_shape_, y_stride, x_stride, grid_stride,
            interpolation_mode_, padding_mode_, align_corners_);
}

void GridSampler2DCpuKernelMod::Call2Half(const float16 *x_data_addr, float16 *y_data_addr,
                                          const float16 *grid_data_addr, std::vector<int64_t> x_dims,
                                          std::vector<int64_t> y_dims, int64_t *y_stride, int64_t *x_stride,
                                          const int64_t *grid_stride, std::string interpolation_mode,
                                          std::string padding_mode, bool align_corners) {
  uint32_t data_num = y_dims[0] * y_dims[2] * y_dims[3];
  auto shard_compute = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int64_t y_iter[4] = {0};
      y_iter[1] = static_cast<int64_t>(i);
      int64_t count = 3;
      while (y_iter[1] > 0) {
        if (count == 1) {
          count--;
        }
        y_iter[count] = y_iter[1] % y_dims[count];
        y_iter[1] /= y_dims[count--];
      }
      const int64_t y_c = y_dims[1];
      int64_t grid_offset = y_iter[0] * grid_stride[0] + y_iter[2] * grid_stride[1] + y_iter[3] * grid_stride[2];
      float x = static_cast<float>(grid_data_addr[grid_offset]);
      float y = static_cast<float>(grid_data_addr[grid_offset + grid_stride[3]]);
      x = GridSamplerComputeSourceIndex<float>(x, x_dims[3], padding_mode, align_corners);
      y = GridSamplerComputeSourceIndex<float>(y, x_dims[2], padding_mode, align_corners);
      auto x_ptr_NC = y_iter[0] * x_stride[0];
      auto y_ptr_NCHW = y_iter[0] * y_stride[0] + y_iter[2] * y_stride[2] + y_iter[3] * y_stride[3];

      if (interpolation_mode == "bilinear") {
        BilinearHalf(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride, x_stride, x_ptr_NC, y_ptr_NCHW);
      } else if (interpolation_mode == "nearest") {
        NearestHalf(x, y, x_data_addr, y_data_addr, y_c, x_dims, y_stride, x_stride, x_ptr_NC, y_ptr_NCHW);
      }
    }
  };
  CPUKernelUtils::ParallelFor(shard_compute, data_num);
}

void GridSampler2DCpuKernelMod::NearestHalf(float x, float y, const float16 *x_data_addr, float16 *y_data_addr,
                                            int64_t y_c, std::vector<int64_t> x_dims, int64_t *y_stride,
                                            const int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
  int64_t x_nearest = static_cast<int64_t>(std::nearbyint(x));
  int64_t y_nearest = static_cast<int64_t>(std::nearbyint(y));
  for (int64_t c = 0; c < y_c; ++c, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    if (WithinBounds2D(y_nearest, x_nearest, x_dims[kTwo], x_dims[kThree])) {
      auto x_index = x_ptr_NC + y_nearest * x_stride[2] + x_nearest * x_stride[3];
      y_data_addr[y_ptr_NCHW] = x_data_addr[x_index];
    } else {
      y_data_addr[y_ptr_NCHW] = static_cast<float16>(0);
    }
  }
}

void GridSampler2DCpuKernelMod::BilinearHalf(float x, float y, const float16 *x_data_addr, float16 *y_data_addr,
                                             int64_t y_c, std::vector<int64_t> x_dims, int64_t *y_stride,
                                             int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW) {
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

  for (int64_t c = 0; c < y_c; ++c, x_ptr_NC += x_stride[1], y_ptr_NCHW += y_stride[1]) {
    y_data_addr[y_ptr_NCHW] = static_cast<float16>(0);
    if (WithinBounds2D(y_tnw, x_tnw, x_dims[kTwo], x_dims[kThree])) {
      auto x_index = x_ptr_NC + y_tnw * x_stride[2] + x_tnw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tnw;
    }
    if (WithinBounds2D(y_tne, x_tne, x_dims[kTwo], x_dims[kThree])) {
      auto x_index = x_ptr_NC + y_tne * x_stride[2] + x_tne * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tne;
    }
    if (WithinBounds2D(y_tsw, x_tsw, x_dims[kTwo], x_dims[kThree])) {
      auto x_index = x_ptr_NC + y_tsw * x_stride[2] + x_tsw * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tsw;
    }
    if (WithinBounds2D(y_tse, x_tse, x_dims[kTwo], x_dims[kThree])) {
      auto x_index = x_ptr_NC + y_tse * x_stride[2] + x_tse * x_stride[3];
      y_data_addr[y_ptr_NCHW] += x_data_addr[x_index] * tse;
    }
  }
}

template <typename T>
void GridSampler2DCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto x_data_addr = static_cast<T *>(inputs[0]->addr);
  auto grid_data_addr = static_cast<T *>(inputs[1]->addr);
  auto output_data_addr = static_cast<T *>(outputs[0]->addr);
  size_t loop_count = LongToSize(output_shape_[0] * output_shape_[2] * output_shape_[3]);
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
  auto x_data_addr = static_cast<float16 *>(inputs[0]->addr);
  auto grid_data_addr = static_cast<float16 *>(inputs[1]->addr);
  auto output_data_addr = static_cast<float16 *>(outputs[0]->addr);
  size_t loop_count = LongToSize(output_shape_[0] * output_shape_[2] * output_shape_[3]);
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
T GridSampler2DCpuKernelMod::GridSamplerComputeSourceIndex(T coord, int64_t size, const std::string &padding_mode,
                                                           bool align_corners) const {
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
T GridSampler2DCpuKernelMod::ReflectCoordinates(T coord, int64_t twice_low, int64_t twice_high) const {
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

bool GridSampler2DCpuKernelMod::WithinBounds2D(int64_t h, int64_t w, int64_t H, int64_t W) const {
  return h >= 0 && h < H && w >= 0 && w < W;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GridSampler2D, GridSampler2DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
