/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/deformable_offsets_kernels.h"
#include <Eigen/Dense>
#include <memory>
#include <map>
#include <functional>
#include <thread>
#include "aicpu_sharder/aicpu_sharder.h"

namespace aicpu {
namespace {
constexpr auto kStrides = "strides";
constexpr auto kPads = "pads";
constexpr auto kSize = "ksize";
constexpr auto kDilations = "dilations";
constexpr auto kModulated = "modulated";
constexpr auto kDeformableGroups = "deformable_groups";
constexpr size_t kInputsSize = 2;
constexpr size_t kOutputsSize = 1;
constexpr size_t kStridesSize = 4;
constexpr size_t kPadsSize = 4;
constexpr size_t kKernelSize = 2;
constexpr size_t kKernelSizeHIndex = 0;
constexpr size_t kKernelSizeWIndex = 1;
constexpr size_t kDilationsSize = 4;
constexpr size_t kXShapeSize = 4;
constexpr size_t kOutputShapeSize = 4;
constexpr size_t kPadTopIndex = 0;
constexpr size_t kPadLeftIndex = 2;
constexpr size_t kOffsetsSize = 3;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
using ShapeVector = std::vector<int64_t>;

template <typename T>
T DeformableBilinear(const T *input, T x, T y, int64_t width, int64_t height) {
  if (y <= static_cast<T>(-1) || y >= static_cast<T>(height) || x <= static_cast<T>(-1) || x >= static_cast<T>(width)) {
    return static_cast<T>(0);
  }
  int64_t left;
  if constexpr (std::is_same<T, float>::value) {
    left = static_cast<int64_t>(floorf(x));
  } else {
    left = static_cast<int64_t>(floor(x));
  }
  auto right = left + 1;
  int64_t top;
  if constexpr (std::is_same<T, float>::value) {
    top = static_cast<int64_t>(floorf(y));
  } else {
    top = static_cast<int64_t>(floor(y));
  }
  auto bottom = top + 1;

  T l = x - static_cast<T>(left);
  T r = static_cast<T>(1) - l;
  T t = y - static_cast<T>(top);
  T b = static_cast<T>(1) - t;

  T lt = static_cast<T>(0);
  T lb = static_cast<T>(0);
  if (left >= 0) {
    if (top >= 0) {
      lt = input[top * width + left];
    }
    if (bottom <= height - 1) {
      lb = input[bottom * width + left];
    }
  }
  T rt = static_cast<T>(0);
  T rb = static_cast<T>(0);
  if (right <= width - 1) {
    if (top >= 0) {
      rt = input[top * width + right];
    }
    if (bottom <= height - 1) {
      rb = input[bottom * width + right];
    }
  }

  T w_lt = r * b;
  T w_rt = l * b;
  T w_lb = r * t;
  T w_rb = l * t;
  T val = (w_lt * lt + w_rt * rt + w_lb * lb + w_rb * rb);
  return val;
}
}  // namespace

uint32_t DeformableOffsetsKernel::ParseAttrs() {
  // Check args.
  n_axis_ = kIndex0;
  c_axis_ = kIndex1;
  h_axis_ = kIndex2;
  w_axis_ = kIndex3;
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();

  auto strides = attrs[kStrides].array();
  if (strides.i_size() != kStridesSize || strides.i(n_axis_) != 1 || strides.i(c_axis_) != 1) {
    AICPU_LOGE(
      "For kernel %s, the strides should be a vector with size %zu and the values according to N and C dimensions must "
      "be set to 1.",
      kernel_name_, kStridesSize);
    return kAicpuKernelStateInvalid;
  }

  auto pads = attrs[kPads].array();
  if (pads.i_size() != kPadsSize) {
    AICPU_LOGE("For kernel %s, the 'pads' should be a vector with size %zu.", kernel_name_, kPadsSize);
    return kAicpuKernelStateInvalid;
  }

  auto kernel_size = attrs[kSize].array();
  if (kernel_size.i_size() != kKernelSize) {
    AICPU_LOGE("For kernel %s, the 'kernel_size' should be a vector with size %zu.", kernel_name_, kKernelSize);
    return kAicpuKernelStateInvalid;
  }

  auto dilations = attrs[kDilations].array();
  if (dilations.i_size() != kDilationsSize || dilations.i(n_axis_) != 1 || dilations.i(c_axis_) != 1) {
    AICPU_LOGE(
      "For kernel %s, the dilations should be a vector with size %zu and the values according to N and C dimensions "
      "must be set to 1.",
      kernel_name_, kStridesSize);
    return kAicpuKernelStateInvalid;
  }

  deformable_groups_ = attrs[kDeformableGroups].i();
  if (deformable_groups_ <= 0) {
    AICPU_LOGE("For kernel %s, the deformable_groups should be greater than 0", kernel_name_);
    return kAicpuKernelStateInvalid;
  }

  for (int32_t i = 0; i < strides.i_size(); i++) {
    strides_.emplace_back(strides.i(i));
  }

  for (int32_t i = 0; i < pads.i_size(); i++) {
    pads_.emplace_back(pads.i(i));
  }

  for (int32_t i = 0; i < kernel_size.i_size(); i++) {
    kernel_size_.emplace_back(kernel_size.i(i));
  }

  for (int32_t i = 0; i < dilations.i_size(); i++) {
    dilations_.emplace_back(dilations.i(i));
  }

  return kAicpuKernelStateSucess;
}

uint32_t DeformableOffsetsKernel::SetDims() {
  auto inputs_shape = node_def_.inputs(kIndex0).tensor_shape();
  if (inputs_shape.dim_size() != kXShapeSize) {
    AICPU_LOGE("For %s, the shape size of input 'x' should be %zu, but got %zu ", kernel_name_, kXShapeSize,
               inputs_shape.dim_size());
    return kAicpuKernelStateInvalid;
  }

  auto outputs_shape = node_def_.outputs(kIndex0).tensor_shape();
  if (outputs_shape.dim_size() != kOutputShapeSize) {
    AICPU_LOGE("For %s, the shape size of output 'y' should be %zu, but got %zu ", kernel_name_, kOutputShapeSize,
               outputs_shape.dim_size());
    return kAicpuKernelStateInvalid;
  }
  ShapeVector x_shape;
  ShapeVector y_shape;
  for (size_t i = 0; i < kXShapeSize; i++) {
    x_shape.emplace_back(inputs_shape.dim(i).size());
    y_shape.emplace_back(outputs_shape.dim(i).size());
  }
  n_ = x_shape[n_axis_];
  c_ = x_shape[c_axis_];
  input_h_ = x_shape[h_axis_];
  input_w_ = x_shape[w_axis_];
  output_h_ = y_shape[h_axis_];
  output_w_ = y_shape[w_axis_];
  position_grid_size_ = output_h_ * output_w_;
  index_type_ = static_cast<aicpuops::DataType>(node_def_.inputs(kIndex0).tensor_type());
  workspace_size_list_.emplace_back(sizeof(int64_t) * LongToSize(position_grid_size_) * kKernelSize);
  return kAicpuKernelStateSucess;
}

uint32_t DeformableOffsetsKernel::ParseKernelParam() {
  auto input_size = node_def_.inputs_size();
  auto output_size = node_def_.outputs_size();
  if (input_size != kInputsSize || output_size != kOutputsSize) {
    AICPU_LOGE("For %s, it should get %zu inputs and %zu outputs, but got %zu input and %zu outputs.", kernel_name_,
               kInputsSize, kOutputsSize, input_size, output_size);
    return kAicpuKernelStateInvalid;
  }
  if (ParseAttrs() != kAicpuKernelStateSucess) {
    return kAicpuKernelStateInvalid;
  }

  if (SetDims() != kAicpuKernelStateSucess) {
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t DeformableOffsetsKernel::DoCompute(const int64_t *position_grid) {
  auto *input_addr = reinterpret_cast<T *>(io_addrs_[kIndex0]);
  auto *offsets_addr = reinterpret_cast<T *>(io_addrs_[kIndex1]);
  auto *output_addr = reinterpret_cast<T *>(io_addrs_[kIndex2]);
  int64_t pixel_h = output_h_ / kernel_size_[kKernelSizeHIndex];
  int64_t pixel_w = output_w_ / kernel_size_[kKernelSizeWIndex];
  int64_t output_c_dim = output_h_ * output_w_;
  int64_t output_n_dim = c_ * output_c_dim;
  int64_t c_size_per_dfm_group = c_ / deformable_groups_;
  int64_t offset_kw_dim = pixel_h * pixel_w;
  int64_t offset_kh_dim = offset_kw_dim * kernel_size_[kKernelSizeWIndex];
  int64_t offset_group_dim = offset_kh_dim * kernel_size_[kKernelSizeHIndex];
  int64_t offset_mask_dim = offset_group_dim * deformable_groups_;
  int64_t offset_n_dim = offset_mask_dim * SizeToLong(kOffsetsSize);
  int64_t input_c_dim = input_h_ * input_w_;
  int64_t input_n_dim = input_c_dim * c_;

  auto task = [this, &input_addr, &offsets_addr, &output_addr, &position_grid, &pixel_w, &output_c_dim, &output_n_dim,
               &c_size_per_dfm_group, &offset_kw_dim, &offset_kh_dim, &offset_group_dim, &offset_mask_dim,
               &offset_n_dim, &input_c_dim, &input_n_dim](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto long_i = SizeToLong(i);
      // Get input position
      int64_t hw_idx = long_i % output_c_dim;
      int64_t position_grid_idx = hw_idx * 2;
      int64_t input_x = position_grid[position_grid_idx];
      int64_t input_y = position_grid[position_grid_idx + 1];
      // Get offsets
      int64_t n_index = long_i / output_n_dim;
      int64_t c_index = long_i / output_c_dim % c_;
      int64_t x = hw_idx % output_w_;
      int64_t y = hw_idx / output_w_;
      int64_t dfm_group_index = c_index / c_size_per_dfm_group;
      int64_t pixel_x = x / kernel_size_[kKernelSizeWIndex];
      int64_t pixel_y = y / kernel_size_[kKernelSizeHIndex];
      int64_t kernel_x = x % kernel_size_[kKernelSizeWIndex];
      int64_t kernel_y = y % kernel_size_[kKernelSizeHIndex];
      int64_t x_offsets_offset = n_index * offset_n_dim + dfm_group_index * offset_group_dim +
                                 kernel_y * offset_kh_dim + kernel_x * offset_kw_dim + pixel_y * pixel_w + pixel_x;
      T x_offsets = offsets_addr[x_offsets_offset];
      int64_t y_offsets_offset = x_offsets_offset + offset_mask_dim;
      T y_offsets = offsets_addr[y_offsets_offset];
      int64_t mask_offset = y_offsets_offset + offset_mask_dim;
      T mask = offsets_addr[mask_offset];

      T new_x = static_cast<T>(input_x) + x_offsets;
      T new_y = static_cast<T>(input_y) + y_offsets;
      const T *input_addr_offset = input_addr + n_index * input_n_dim + c_index * input_c_dim;
      T bilinear_val = DeformableBilinear(input_addr_offset, new_x, new_y, input_w_, input_h_);
      output_addr[i] = bilinear_val * mask;
    }
  };
  int64_t num_kernels = n_ * output_n_dim;
  int64_t per_unit_size = num_kernels / std::thread::hardware_concurrency();
  ParallelFor(UlongToLong(num_kernels), per_unit_size, task);
  return kAicpuKernelStateSucess;
}

uint32_t DeformableOffsetsKernel::GenPositionGrid(int64_t *position_grid) {
  auto task = [this, &position_grid](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto long_i = SizeToLong(i);
      int64_t y = long_i / output_w_;
      int64_t x = long_i % output_w_;
      int64_t pixel_y = y / kernel_size_[kKernelSizeHIndex];
      int64_t pixel_x = x / kernel_size_[kKernelSizeWIndex];
      int64_t kernel_y = y % kernel_size_[kKernelSizeHIndex];
      int64_t kernel_x = x % kernel_size_[kKernelSizeWIndex];
      size_t index = i * 2;
      position_grid[index] = pixel_x * strides_[w_axis_] + kernel_x * dilations_[w_axis_] - pads_[kPadLeftIndex];
      position_grid[index + 1] = pixel_y * strides_[h_axis_] + kernel_y * dilations_[h_axis_] - pads_[kPadTopIndex];
    }
  };

  int64_t num_kernels = output_h_ * output_w_;
  int64_t per_unit_size = num_kernels / std::thread::hardware_concurrency();
  ParallelFor(UlongToLong(num_kernels), per_unit_size, task);
  return kAicpuKernelStateSucess;
}

uint32_t DeformableOffsetsKernel::DoCompute() {
  auto *position_grid_addr = reinterpret_cast<int64_t *>(malloc(workspace_size_list_[0]));
  if (position_grid_addr == nullptr) {
    AICPU_LOGE("Malloc memory failed!");
    return kAicpuKernelStateInternalError;
  }
  auto ret = GenPositionGrid(position_grid_addr);
  if (ret != kAicpuKernelStateSucess) {
    AICPU_LOGE("Generate position grid failed.");
    free(position_grid_addr);
    return kAicpuKernelStateFailed;
  }
  switch (index_type_) {
    case aicpuops::DataType::MS_FLOAT32:
      ret = DoCompute<float>(position_grid_addr);
      break;
    case aicpuops::DataType::MS_FLOAT16:
      ret = DoCompute<Eigen::half>(position_grid_addr);
      break;
    default:
      AICPU_LOGE("Error type %s.", typeid(index_type_).name());
      free(position_grid_addr);
      return kAicpuKernelStateFailed;
  }
  free(position_grid_addr);
  return ret;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t DeformableOffsets(void *param) {
  aicpu::DeformableOffsetsKernel deformable_offsets_kernel;
  return deformable_offsets_kernel.Compute(param);
}
}  // namespace kernel
