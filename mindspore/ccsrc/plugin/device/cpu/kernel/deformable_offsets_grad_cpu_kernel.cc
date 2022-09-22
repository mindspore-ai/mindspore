/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/deformable_offsets_grad_cpu_kernel.h"
#include <utility>
#include <set>
#include <map>
#include <functional>
#include <tuple>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/deformable_offsets_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 2;

constexpr size_t kGradIndex = 0;
constexpr size_t kXIndex = 1;
constexpr size_t kOffsetIndex = 2;
constexpr size_t kGradXIndex = 0;
constexpr size_t kGradOffsetIndex = 1;

constexpr size_t kPadNum = 4;
constexpr size_t kStrideNum = 4;
constexpr size_t kDilationNum = 4;
constexpr size_t kKernelSizeNum = 2;

constexpr size_t kPadTopIndex = 0;
constexpr size_t kPadLeftIndex = 2;
constexpr size_t kStrideHIndex = 2;
constexpr size_t kStrideWIndex = 3;
constexpr size_t kDilationHIndex = 2;
constexpr size_t kDilationWIndex = 3;
constexpr size_t kKernelHIndex = 0;
constexpr size_t kKernelWIndex = 1;

constexpr size_t kCIndexForNCHW = 1;
constexpr size_t kHIndexForNCHW = 2;
constexpr size_t kWIndexForNCHW = 3;

constexpr size_t kHIndexForNHWC = 1;
constexpr size_t kWIndexForNHWC = 2;
constexpr size_t kCIndexForNHWC = 3;

// x,y,mask total occupy 3 channel
constexpr size_t kOffsetChannel = 3;
struct OffsetStride {
  size_t kernel_w_stride;
  size_t kernel_h_stride;
  size_t deformable_group_stride;
  size_t position_stride;
  size_t offset_w_stride;
  size_t offset_h_stride;
  size_t n_stride;
};

struct GradStride {
  size_t deformable_group_channel_stride;
  size_t deformable_group_stride;
  size_t kernel_w_stride;
  size_t offset_w_stride;
  size_t kernel_h_stride;
  size_t offset_h_stride;
  size_t n_stride;
};

struct InputXStride {
  size_t deformable_group_channel_stride;
  size_t deformable_group_stride;
  size_t w_stride;
  size_t h_stride;
  size_t n_stride;
};

struct OffsetIndex {
  size_t kernel_j;
  size_t kernel_i;
  size_t deformable_group_i;
  size_t offset_j;
  size_t offset_i;
  size_t n_i;

  std::string ToString() const {
    std::ostringstream buffer;
    buffer << "offset_j:" << offset_j << "\n";
    buffer << "offset_i:" << offset_i << "\n";
    buffer << "kernel_j:" << kernel_j << "\n";
    buffer << "kernel_i:" << kernel_i << "\n";
    buffer << "deformable_group_i:" << deformable_group_i << "\n";
    buffer << "n_i:" << n_i << "\n";
    return buffer.str();
  }
};

struct InputXIndex {
  float i;
  float j;
};
}  // namespace

std::mutex mutex_;

template <typename T>
void MsAtomicAdd(T *output_grad_x, const size_t &output_grad_base_pos, const T &added_value) {
  std::lock_guard<std::mutex> lock(mutex_);
  output_grad_x[output_grad_base_pos] += added_value;
}

inline std::tuple<size_t, size_t, size_t> CalPosition(const OffsetIndex &offset_index,
                                                      const OffsetStride &offset_stride, const GradStride &grad_stride,
                                                      const InputXStride &input_x_stride) {
  const size_t offset_index_base_pos =
    offset_index.n_i * offset_stride.n_stride +
    offset_index.deformable_group_i * offset_stride.deformable_group_stride +
    offset_index.kernel_i * offset_stride.kernel_h_stride + offset_index.kernel_j * offset_stride.kernel_w_stride +
    offset_index.offset_i * offset_stride.offset_h_stride + offset_index.offset_j * offset_stride.offset_w_stride;

  const size_t input_grad_base_pos =
    offset_index.n_i * grad_stride.n_stride + offset_index.offset_i * grad_stride.offset_h_stride +
    offset_index.offset_j * grad_stride.offset_w_stride + offset_index.kernel_i * grad_stride.kernel_h_stride +
    offset_index.kernel_j * grad_stride.kernel_w_stride +
    offset_index.deformable_group_i * grad_stride.deformable_group_stride;
  const size_t input_x_base_pos = offset_index.n_i * input_x_stride.n_stride +
                                  offset_index.deformable_group_i * input_x_stride.deformable_group_stride;
  return {offset_index_base_pos, input_grad_base_pos, input_x_base_pos};
}

inline InputXIndex CalInputXIndex(const OffsetIndex &offset_index, const DeformableOffsetGradDims &dims) {
  InputXIndex input_x_index;
  input_x_index.i = -1.0f * SizeToFloat(dims.pad_top);
  input_x_index.j = -1.0f * SizeToFloat(dims.pad_left);
  input_x_index.i += SizeToFloat(offset_index.offset_i * dims.stride_h + offset_index.kernel_i * dims.dilation_h);
  input_x_index.j += SizeToFloat(offset_index.offset_j * dims.stride_w + offset_index.kernel_j * dims.dilation_w);
  return input_x_index;
}

template <typename T>
void DeformableOffsetGradKernel(const OffsetStride &offset_stride, const GradStride &grad_stride,
                                const DeformableOffsetGradDims &dims, const InputXIndex &input_x_index,
                                const InputXStride &input_x_stride, const size_t &offset_index_base_pos,
                                const size_t &input_grad_base_pos, const size_t &input_x_base_pos, const T *input_x,
                                const T *input_offset, const T *input_grad, T *output_grad_x, T *output_grad_offset) {
  const size_t offset_index_i = offset_index_base_pos + offset_stride.position_stride;
  const size_t offset_index_weight = offset_index_base_pos + 2 * offset_stride.position_stride;
  float offset_i = static_cast<float>(input_offset[offset_index_i]);
  float offset_j = static_cast<float>(input_offset[offset_index_base_pos]);
  float scale_weight = static_cast<float>(input_offset[offset_index_weight]);

  float floor_offset_i = floorf(offset_i);
  float floor_offset_j = floorf(offset_j);
  float ceil_offset_i = floor_offset_i + 1;
  float ceil_offset_j = floor_offset_j + 1;

  float floor_i = input_x_index.i + floor_offset_i;
  float floor_j = input_x_index.j + floor_offset_j;
  float ceil_i = input_x_index.i + ceil_offset_i;
  float ceil_j = input_x_index.j + ceil_offset_j;

  float ceil_weight_i = offset_i + 1 - ceil_offset_i;
  float ceil_weight_j = offset_j + 1 - ceil_offset_j;
  float floor_weight_i = 1 - ceil_weight_i;
  float floor_weight_j = 1 - ceil_weight_j;

  float floor_floor_weight = floor_weight_i * floor_weight_j;
  float ceil_floor_weight = ceil_weight_i * floor_weight_j;
  float floor_ceil_weight = floor_weight_i * ceil_weight_j;
  float ceil_ceil_weight = ceil_weight_i * ceil_weight_j;

  bool floor_floor_valid = false;
  bool ceil_floor_valid = false;
  bool floor_ceil_valid = false;
  bool ceil_ceil_valid = false;
  if (floor_i >= 0 && floor_i < dims.x_h) {
    if (floor_j >= 0 && floor_j < dims.x_w) {
      floor_floor_valid = true;
    }
    if (ceil_j >= 0 && ceil_j < dims.x_w) {
      floor_ceil_valid = true;
    }
  }

  if (ceil_i >= 0 && ceil_i < dims.x_h) {
    if (floor_j >= 0 && floor_j < dims.x_w) {
      ceil_floor_valid = true;
    }
    if (ceil_j >= 0 && ceil_j < dims.x_w) {
      ceil_ceil_valid = true;
    }
  }

  for (size_t channel = 0; channel < dims.deformable_group_channel; ++channel) {
    float grad =
      static_cast<float>(input_grad[input_grad_base_pos + channel * grad_stride.deformable_group_channel_stride]);
    float grad_scale = grad * scale_weight;
    size_t tmp_input_x_base_pos = input_x_base_pos + channel * input_x_stride.deformable_group_channel_stride;
    float current_x_pos;
    float floor_floor_value = 0;
    float ceil_floor_value = 0;
    float floor_ceil_value = 0;
    float ceil_ceil_value = 0;
    size_t input_x_pos = 0;
    if (floor_floor_valid) {
      current_x_pos = tmp_input_x_base_pos + floor_i * input_x_stride.h_stride + floor_j * input_x_stride.w_stride;
      input_x_pos = static_cast<size_t>(current_x_pos);
      floor_floor_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x, input_x_pos, static_cast<T>(grad_scale * floor_floor_weight));
    }

    if (ceil_floor_valid) {
      current_x_pos = tmp_input_x_base_pos + ceil_i * input_x_stride.h_stride + floor_j * input_x_stride.w_stride;
      input_x_pos = static_cast<size_t>(current_x_pos);
      ceil_floor_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x, input_x_pos, static_cast<T>(grad_scale * ceil_floor_weight));
    }

    if (floor_ceil_valid) {
      current_x_pos = tmp_input_x_base_pos + floor_i * input_x_stride.h_stride + ceil_j * input_x_stride.w_stride;
      input_x_pos = static_cast<size_t>(current_x_pos);
      floor_ceil_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x, input_x_pos, static_cast<T>(grad_scale * floor_ceil_weight));
    }

    if (ceil_ceil_valid) {
      current_x_pos = tmp_input_x_base_pos + ceil_i * input_x_stride.h_stride + ceil_j * input_x_stride.w_stride;
      input_x_pos = static_cast<size_t>(current_x_pos);
      ceil_ceil_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x, input_x_pos, static_cast<T>(grad_scale * ceil_ceil_weight));
    }

    float delta = -floor_floor_value * floor_weight_j + ceil_floor_value * floor_weight_j -
                  floor_ceil_value * ceil_weight_j + ceil_ceil_value * ceil_weight_j;
    delta *= grad_scale;
    output_grad_offset[offset_index_i] += static_cast<T>(delta);

    delta = -floor_floor_value * floor_weight_i - ceil_floor_value * ceil_weight_i + floor_ceil_value * floor_weight_i +
            ceil_ceil_value * ceil_weight_i;
    delta *= grad_scale;
    output_grad_offset[offset_index_base_pos] += static_cast<T>(delta);

    delta = floor_floor_value * floor_floor_weight + ceil_floor_value * ceil_floor_weight +
            floor_ceil_value * floor_ceil_weight + ceil_ceil_value * ceil_ceil_weight;
    delta *= grad;
    output_grad_offset[offset_index_weight] += static_cast<T>(delta);
  }
}

template <typename T>
void DeformableOffsetsGradCpuKernelMod::DeformableOffsetGradNHWCKernel(size_t num_kernels,
                                                                       const DeformableOffsetGradDims &dims,
                                                                       const T *input_x, const T *input_offset,
                                                                       const T *input_grad, T *output_grad_x,
                                                                       T *output_grad_offset) {
  OffsetStride offset_stride;
  offset_stride.kernel_w_stride = 1;
  offset_stride.kernel_h_stride = dims.kernel_w * offset_stride.kernel_w_stride;
  offset_stride.deformable_group_stride = dims.kernel_h * offset_stride.kernel_h_stride;
  offset_stride.position_stride = dims.deformable_group * offset_stride.deformable_group_stride;
  offset_stride.offset_w_stride = kOffsetChannel * offset_stride.position_stride;
  offset_stride.offset_h_stride = dims.offset_w * offset_stride.offset_w_stride;
  offset_stride.n_stride = dims.offset_h * offset_stride.offset_h_stride;

  GradStride grad_stride;
  grad_stride.deformable_group_channel_stride = 1;
  grad_stride.deformable_group_stride = dims.deformable_group_channel * grad_stride.deformable_group_channel_stride;
  grad_stride.kernel_w_stride = dims.deformable_group * grad_stride.deformable_group_stride;
  grad_stride.offset_w_stride = dims.kernel_w * grad_stride.kernel_w_stride;
  grad_stride.kernel_h_stride = dims.offset_w * grad_stride.offset_w_stride;
  grad_stride.offset_h_stride = dims.kernel_h * grad_stride.kernel_h_stride;
  grad_stride.n_stride = dims.offset_h * grad_stride.offset_h_stride;

  InputXStride input_x_stride;
  input_x_stride.deformable_group_channel_stride = 1;
  input_x_stride.deformable_group_stride =
    dims.deformable_group_channel * input_x_stride.deformable_group_channel_stride;
  input_x_stride.w_stride = dims.deformable_group * input_x_stride.deformable_group_stride;
  input_x_stride.h_stride = dims.x_w * input_x_stride.w_stride;
  input_x_stride.n_stride = dims.x_h * input_x_stride.h_stride;

  OffsetIndex offset_index;
  auto task = [&offset_index, &dims, &offset_stride, &grad_stride, &input_x, &input_offset, &input_grad, &output_grad_x,
               &output_grad_offset, &input_x_stride](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      offset_index.kernel_j = index % dims.kernel_w;
      size_t tmp = index / dims.kernel_w;
      offset_index.kernel_i = tmp % dims.kernel_h;
      tmp = tmp / dims.kernel_h;
      offset_index.deformable_group_i = tmp % dims.deformable_group;
      tmp = tmp / dims.deformable_group;
      offset_index.offset_j = tmp % dims.offset_w;
      tmp = tmp / dims.offset_w;
      offset_index.offset_i = tmp % dims.offset_h;
      offset_index.n_i = tmp / dims.offset_h;

      const auto [offset_index_base_pos, input_grad_base_pos, input_x_base_pos] =
        CalPosition(offset_index, offset_stride, grad_stride, input_x_stride);
      const auto input_x_index = CalInputXIndex(offset_index, dims);
      DeformableOffsetGradKernel(offset_stride, grad_stride, dims, input_x_index, input_x_stride, offset_index_base_pos,
                                 input_grad_base_pos, input_x_base_pos, input_x, input_offset, input_grad,
                                 output_grad_x, output_grad_offset);
    }
  };
  ParallelLaunchAutoSearch(task, num_kernels, this, &parallel_search_info_, pool_);
}

template <typename T>
void DeformableOffsetsGradCpuKernelMod::DeformableOffsetGradNCHWKernel(size_t num_kernels,
                                                                       const DeformableOffsetGradDims &dims,
                                                                       const T *input_x, const T *input_offset,
                                                                       const T *input_grad, T *output_grad_x,
                                                                       T *output_grad_offset) {
  OffsetStride offset_stride;
  offset_stride.offset_w_stride = 1;
  offset_stride.offset_h_stride = dims.offset_w * offset_stride.offset_w_stride;
  offset_stride.kernel_w_stride = dims.offset_h * offset_stride.offset_h_stride;
  offset_stride.kernel_h_stride = dims.kernel_w * offset_stride.kernel_w_stride;
  offset_stride.deformable_group_stride = dims.kernel_h * offset_stride.kernel_h_stride;
  offset_stride.position_stride = dims.deformable_group * offset_stride.deformable_group_stride;
  offset_stride.n_stride = kOffsetChannel * offset_stride.position_stride;

  GradStride grad_stride;
  grad_stride.kernel_w_stride = 1;
  grad_stride.offset_w_stride = dims.kernel_w * grad_stride.kernel_w_stride;
  grad_stride.kernel_h_stride = dims.offset_w * grad_stride.offset_w_stride;
  grad_stride.offset_h_stride = dims.kernel_h * grad_stride.kernel_h_stride;
  grad_stride.deformable_group_channel_stride = dims.offset_h * grad_stride.offset_h_stride;
  grad_stride.deformable_group_stride = dims.deformable_group_channel * grad_stride.deformable_group_channel_stride;
  grad_stride.n_stride = dims.deformable_group * grad_stride.deformable_group_stride;

  InputXStride input_x_stride;
  input_x_stride.w_stride = 1;
  input_x_stride.h_stride = dims.x_w * input_x_stride.w_stride;
  input_x_stride.deformable_group_channel_stride = dims.x_h * input_x_stride.h_stride;
  input_x_stride.deformable_group_stride =
    dims.deformable_group_channel * input_x_stride.deformable_group_channel_stride;
  input_x_stride.n_stride = dims.deformable_group * input_x_stride.deformable_group_stride;

  OffsetIndex offset_index;

  auto task = [&offset_index, &dims, &offset_stride, &grad_stride, &input_x, &input_offset, &input_grad, &output_grad_x,
               &output_grad_offset, &input_x_stride](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      offset_index.offset_j = index % dims.offset_w;
      size_t tmp = index / dims.offset_w;
      offset_index.offset_i = tmp % dims.offset_h;
      tmp = tmp / dims.offset_h;
      offset_index.kernel_j = tmp % dims.kernel_w;
      tmp = tmp / dims.kernel_w;
      offset_index.kernel_i = tmp % dims.kernel_h;
      tmp = tmp / dims.kernel_h;
      offset_index.deformable_group_i = tmp % dims.deformable_group;
      offset_index.n_i = tmp / dims.deformable_group;

      const auto [offset_index_base_pos, input_grad_base_pos, input_x_base_pos] =
        CalPosition(offset_index, offset_stride, grad_stride, input_x_stride);
      const auto input_x_index = CalInputXIndex(offset_index, dims);
      DeformableOffsetGradKernel(offset_stride, grad_stride, dims, input_x_index, input_x_stride, offset_index_base_pos,
                                 input_grad_base_pos, input_x_base_pos, input_x, input_offset, input_grad,
                                 output_grad_x, output_grad_offset);
    }
  };
  ParallelLaunchAutoSearch(task, num_kernels, this, &parallel_search_info_, pool_);
}

bool DeformableOffsetsGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  deformable_kernel_operator_ = std::make_shared<ops::DeformableOffsetsGrad>(base_operator->GetPrim());

  CheckInOutNum(inputs.size(), outputs.size());
  auto match_result = MatchKernelFunc(base_operator, inputs, outputs);
  if (!match_result) {
    return false;
  }
  GetDataFormat();
  return true;
}

bool DeformableOffsetsGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &workspace,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool DeformableOffsetsGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  const size_t num_kernels =
    dims_.x_n * dims_.offset_h * dims_.offset_w * dims_.kernel_h * dims_.kernel_w * dims_.deformable_group;
  const T *input_grad = GetDeviceAddress<T>(inputs, kGradXIndex);
  const T *input_x = GetDeviceAddress<T>(inputs, kXIndex);
  const T *input_offset = GetDeviceAddress<T>(inputs, kOffsetIndex);
  T *output_grad_x = GetDeviceAddress<T>(outputs, kGradXIndex);
  T *output_grad_offset = GetDeviceAddress<T>(outputs, kGradOffsetIndex);

  auto grad_x_size = outputs[kGradXIndex]->size;
  auto grad_offset_size = outputs[kGradOffsetIndex]->size;
  // Reset output initial value to 0.
  auto ret = memset_s(output_grad_x, grad_x_size, 0, grad_x_size);
  if (ret != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output_grad_x  memset_s error. Error no: " << ret;
  }
  ret = memset_s(output_grad_offset, grad_offset_size, 0, grad_offset_size);
  if (ret != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output_grad_offset memset_s error. Error no: " << ret;
  }
  if (data_format_ == kOpFormat_NCHW) {
    DeformableOffsetGradNCHWKernel<T>(num_kernels, dims_, input_x, input_offset, input_grad, output_grad_x,
                                      output_grad_offset);
  } else {
    DeformableOffsetGradNHWCKernel<T>(num_kernels, dims_, input_x, input_offset, input_grad, output_grad_x,
                                      output_grad_offset);
  }
  return true;
}

void DeformableOffsetsGradCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

int DeformableOffsetsGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (input_size_list_.size() != kInputNum || output_size_list_.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << " resize : input and output size should be " << kInputNum << " and " << kOutputNum
                  << ", but get " << input_size_list_.size() << " and " << output_size_list_.size();
    return KRET_RESIZE_FAILED;
  }
  SetDims(base_operator, inputs);
  return KRET_OK;
}

using KernelAttrAndFuncList =
  const std::vector<std::pair<KernelAttr, DeformableOffsetsGradCpuKernelMod::KernelRunFunc>>;
KernelAttrAndFuncList &DeformableOffsetsGradCpuKernelMod::GetFuncList() const {
  static KernelAttrAndFuncList kernel_attr_and_func_list = {{KernelAttr()
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             &DeformableOffsetsGradCpuKernelMod::LaunchKernel<float16>},
                                                            {KernelAttr()
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             &DeformableOffsetsGradCpuKernelMod::LaunchKernel<float>}};

  return kernel_attr_and_func_list;
}  // namespace kernel

void DeformableOffsetsGradCpuKernelMod::GetDataFormat() { data_format_ = deformable_kernel_operator_->get_format(); }

void DeformableOffsetsGradCpuKernelMod::CheckInOutNum(size_t inputs_num, size_t outputs_num) const {
  if (inputs_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kInputNum << ", but got "
                      << inputs_num;
  }
  if (outputs_num != kOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be " << kOutputNum << ", but got "
                      << outputs_num;
  }
}

void DeformableOffsetsGradCpuKernelMod::SetDims(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DeformableOffsetsGrad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Cast DeformableOffsetsGrad failed!";
  }
  dims_.deformable_group = LongToSize(kernel_ptr->get_deformable_groups());
  if (dims_.deformable_group == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', deformable group must be greater than 0.";
  }
  std::vector<int64_t> pad = kernel_ptr->get_pads();
  if (pad.size() != kPadNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be " << kPadNum << ", but got "
                      << pad.size();
  }
  dims_.pad_top = LongToSize(pad[kPadTopIndex]);
  dims_.pad_left = LongToSize(pad[kPadLeftIndex]);

  std::vector<int64_t> stride = kernel_ptr->get_strides();
  if (stride.size() != kStrideNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be " << kStrideNum << ", but got "
                      << stride.size();
  }
  dims_.stride_h = LongToSize(stride[kStrideHIndex]);
  dims_.stride_w = LongToSize(stride[kStrideWIndex]);

  std::vector<int64_t> dilation = kernel_ptr->get_dilations();
  if (dilation.size() != kDilationNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be " << kDilationNum
                      << ", but got " << dilation.size();
  }
  dims_.dilation_h = LongToSize(dilation[kDilationHIndex]);
  dims_.dilation_w = LongToSize(dilation[kDilationWIndex]);

  std::vector<int64_t> ksize = kernel_ptr->get_kernel_size();
  if (ksize.size() != kKernelSizeNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'ksize' must be " << kKernelSizeNum
                      << ", but got " << ksize.size();
  }
  dims_.kernel_h = LongToSize(ksize[kKernelHIndex]);
  dims_.kernel_w = LongToSize(ksize[kKernelWIndex]);
  if (dims_.kernel_h == 0 || dims_.kernel_w == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'ksize' must be larger than 0.";
  }
  auto x_shape = inputs[kXIndex]->GetShapeVector();
  dims_.x_n = LongToSize(x_shape[0]);

  auto grad_shape = inputs[kGradIndex]->GetShapeVector();
  if (data_format_ == kOpFormat_NCHW) {
    dims_.grad_h = LongToSize(grad_shape[kHIndexForNCHW]);
    dims_.grad_w = LongToSize(grad_shape[kWIndexForNCHW]);
    dims_.x_h = LongToSize(x_shape[kHIndexForNCHW]);
    dims_.x_w = LongToSize(x_shape[kWIndexForNCHW]);
    dims_.deformable_group_channel = LongToSize(x_shape[kCIndexForNCHW]) / dims_.deformable_group;
  } else {
    dims_.grad_h = LongToSize(grad_shape[kHIndexForNHWC]);
    dims_.grad_w = LongToSize(grad_shape[kWIndexForNHWC]);
    dims_.x_h = LongToSize(x_shape[kHIndexForNHWC]);
    dims_.x_w = LongToSize(x_shape[kWIndexForNHWC]);
    dims_.deformable_group_channel = LongToSize(x_shape[kCIndexForNHWC]) / dims_.deformable_group;
  }
  dims_.offset_h = dims_.grad_h / dims_.kernel_h;
  dims_.offset_w = dims_.grad_w / dims_.kernel_w;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DeformableOffsetsGrad, DeformableOffsetsGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
