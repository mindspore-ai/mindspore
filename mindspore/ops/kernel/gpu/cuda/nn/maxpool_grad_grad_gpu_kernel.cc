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

#include "kernel/gpu/cuda/nn/maxpool_grad_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <string>
#include "mindspore/ops/infer/grad/max_pool_grad_grad.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/maxpool_grad_grad_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradGradInputsNum = 3;
constexpr size_t kMaxPoolGradGradOutputsNum = 1;
constexpr size_t kGradIndex = 2;
constexpr size_t kPadHalf = 2;
}  // namespace

bool MaxPoolGradGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMaxPoolGradGradInputsNum || outputs.size() != kMaxPoolGradGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolGradGradInputsNum << " and "
                  << kMaxPoolGradGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  kernels_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(ops::kKernelSize));
  strides_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(ops::kStrides));
  auto mode_str = GetValue<std::string>(primitive_->GetAttr(ops::kPadMode));
  (void)std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
  MS_EXCEPTION_IF_CHECK_FAIL((mode_str == "SAME" || mode_str == "VALID"),
                             "MaxPoolGradGrad only supports pad mode same or valid, but get " + mode_str);
  pad_mode_ = mode_str == "SAME" ? PadMode::SAME : PadMode::VALID;
  if (pad_mode_ != PadMode::SAME && pad_mode_ != PadMode::VALID) {
    MS_LOG(ERROR) << kernel_name_ << " only support pad mode same or valid, but get " << pad_mode_;
    return false;
  }

  depth_index_ = (dim_ == kMaxPool2DGradGradDim) ? 0 : kDim2;
  height_index_ = (dim_ == kMaxPool2DGradGradDim) ? kDim2 : kDim3;
  width_index_ = (dim_ == kMaxPool2DGradGradDim) ? kDim3 : kDim4;

  window_height_ = LongToInt(kernels_[height_index_]);
  window_width_ = LongToInt(kernels_[width_index_]);
  stride_height_ = LongToInt(strides_[height_index_]);
  stride_width_ = LongToInt(strides_[width_index_]);
  if (dim_ == kMaxPool3DGradGradDim) {
    window_depth_ = LongToInt(kernels_[depth_index_]);
    stride_depth_ = LongToInt(strides_[depth_index_]);
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void MaxPoolGradGradGpuKernelMod::CalPad() {
  if (pad_mode_ == PadMode::VALID) {
    pad_front_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    return;
  }

  std::vector<int64_t> pad(in_shapes_.size(), 0);
  for (int i = 0; i < dim_; i++) {
    auto cur_dim = i + 2;
    MS_EXCEPTION_IF_ZERO("stride ", strides_[cur_dim]);
    auto tmp_dim_size = (in_shapes_[cur_dim] / strides_[cur_dim]) * strides_[cur_dim] == in_shapes_[cur_dim]
                          ? (in_shapes_[cur_dim] / strides_[cur_dim])
                          : (in_shapes_[cur_dim] / strides_[cur_dim]) + 1;
    auto pad_t = std::max<int>(0, (tmp_dim_size - 1) * strides_[cur_dim] + kernels_[cur_dim] - in_shapes_[cur_dim]);
    pad[cur_dim] = pad_t / kPadHalf;
  }

  pad_top_ = pad[height_index_];
  pad_left_ = pad[width_index_];
  if (dim_ == kMaxPool3DGradGradDim) {
    pad_front_ = pad[depth_index_];
  }
}

int MaxPoolGradGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  in_shapes_ = inputs[kIndex0]->GetShapeVector();
  batch_ = LongToInt(in_shapes_[kDim0]);
  channel_ = LongToInt(in_shapes_[kDim1]);
  input_depth_ = LongToInt(in_shapes_[depth_index_]);
  input_height_ = LongToInt(in_shapes_[height_index_]);
  input_width_ = LongToInt(in_shapes_[width_index_]);
  input_batch_stride_ = std::accumulate(in_shapes_.begin() + 1, in_shapes_.end(), 1, std::multiplies<size_t>());

  out_shapes_ = outputs[kIndex0]->GetShapeVector();
  output_depth_ = LongToInt(out_shapes_[depth_index_]);
  output_height_ = LongToInt(out_shapes_[height_index_]);
  output_width_ = LongToInt(out_shapes_[width_index_]);
  output_batch_stride_ = std::accumulate(out_shapes_.begin() + 1, out_shapes_.end(), 1, std::multiplies<size_t>());

  CalPad();
  return KRET_OK;
}

template <typename T>
bool MaxPoolGradGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *grad_addr = GetDeviceAddress<T>(inputs, kGradIndex);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (dim_ == kMaxPool2DGradGradDim) {
    CalMaxPoolGradGrad<T>(input_addr, grad_addr, batch_, channel_, input_height_, input_width_, window_height_,
                          window_width_, stride_height_, stride_width_, pad_top_, pad_left_, output_height_,
                          output_width_, output_addr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else if (dim_ == kMaxPool3DGradGradDim) {
    CalMaxPool3DGradGrad<T>(input_addr, grad_addr, batch_, channel_, input_depth_, input_height_, input_width_,
                            window_depth_, window_height_, window_width_, stride_depth_, stride_height_, stride_width_,
                            pad_front_, pad_top_, pad_left_, output_depth_, output_height_, output_width_, output_addr,
                            device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', only supports 2D or 3D max pooling.";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaxPoolGradGradGpuKernelMod::MaxPoolGradGradFunc>>
  MaxPoolGradGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MaxPoolGradGradGpuKernelMod::LaunchKernel<half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MaxPoolGradGradGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> MaxPoolGradGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolGradGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolGradGrad, MaxPool2DGradGradGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPool3DGradGrad, MaxPool3DGradGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
