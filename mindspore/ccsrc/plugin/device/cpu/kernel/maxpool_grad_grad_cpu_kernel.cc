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

#include "plugin/device/cpu/kernel/maxpool_grad_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "utils/ms_utils.h"
#include "utils/profile.h"
#include "mindspore/ccsrc/kernel/common_utils.h"
#include "nnacl/fp32_grad/maxpool_grad_grad.h"
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradGradInputsNum = 3;
constexpr size_t kMaxPoolGradGradOutputsNum = 1;
constexpr size_t kGradIndex = 2;
constexpr int64_t kPadHalf = 2;

const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)},
};
}  // namespace

bool MaxPoolGradGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kMaxPoolGradGradInputsNum || outputs.size() != kMaxPoolGradGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolGradGradInputsNum << " and "
                  << kMaxPoolGradGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast MaxPoolGradGrad ops failed!";
    return false;
  }
  kernels_ = kernel_ptr->get_kernel_size();
  strides_ = kernel_ptr->get_strides();
  pad_mode_ = kernel_ptr->get_pad_mode();
  if (pad_mode_ != PadMode::SAME && pad_mode_ != PadMode::VALID) {
    MS_LOG(ERROR) << kernel_name_ << " only support pad mode same or valid, but get " << pad_mode_;
    return false;
  }

  depth_index_ = (dim_ == kMaxPool2DGradGradDim) ? 0 : kDim2;
  height_index_ = (dim_ == kMaxPool2DGradGradDim) ? kDim2 : kDim3;
  width_index_ = (dim_ == kMaxPool2DGradGradDim) ? kDim3 : kDim4;

  param_ = (dim_ == kMaxPool2DGradGradDim) ? reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)))
                                           : reinterpret_cast<PoolingParameter *>(malloc(sizeof(Pooling3DParameter)));
  MS_ERROR_IF_NULL(param_);
  param_->window_h_ = LongToInt(kernels_[height_index_]);
  param_->window_w_ = LongToInt(kernels_[width_index_]);
  param_->stride_h_ = LongToInt(strides_[height_index_]);
  param_->stride_w_ = LongToInt(strides_[width_index_]);
  if (dim_ == kMaxPool3DGradGradDim) {
    reinterpret_cast<Pooling3DParameter *>(param_)->window_d_ = LongToInt(kernels_[depth_index_]);
    reinterpret_cast<Pooling3DParameter *>(param_)->stride_d_ = LongToInt(strides_[depth_index_]);
  }
  return true;
}

void MaxPoolGradGradCpuKernelMod::CheckInputVaild() const {
  const size_t src_dim = in_shapes_.size();
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "PoolingGrad only supports 4D/5D input, but got " << src_dim << "D";
  }
  if (kernels_.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires kernel_size must be " << src_dim << "D, but got " << kernels_.size()
                      << "D!";
  }
  if (strides_.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires strides must be " << src_dim << "D, but got " << strides_.size()
                      << "D!";
  }
}

void MaxPoolGradGradCpuKernelMod::CalPad() {
  if (pad_mode_ == PadMode::VALID) {
    if (dim_ == kMaxPool3DGradGradDim) {
      reinterpret_cast<Pooling3DParameter *>(param_)->pad_f_ = 0;
    }
    param_->pad_u_ = 0;
    param_->pad_l_ = 0;
    return;
  }

  std::vector<int64_t> pad(in_shapes_.size(), 0);
  for (size_t i = 0; i < dim_; i++) {
    size_t cur_dim = i + 2;
    MS_EXCEPTION_IF_ZERO("stride ", strides_[cur_dim]);
    int64_t tmp_dim_size = (in_shapes_[cur_dim] / strides_[cur_dim]) * strides_[cur_dim] == in_shapes_[cur_dim]
                             ? (in_shapes_[cur_dim] / strides_[cur_dim])
                             : (in_shapes_[cur_dim] / strides_[cur_dim]) + 1;
    int64_t pad_t =
      std::max<int64_t>(0, (tmp_dim_size - 1) * strides_[cur_dim] + kernels_[cur_dim] - in_shapes_[cur_dim]);
    pad[cur_dim] = pad_t / kPadHalf;
  }

  param_->pad_u_ = LongToInt(pad[height_index_]);
  param_->pad_l_ = LongToInt(pad[width_index_]);
  if (dim_ == kMaxPool3DGradGradDim) {
    reinterpret_cast<Pooling3DParameter *>(param_)->pad_f_ = LongToInt(pad[depth_index_]);
  }
}

int MaxPoolGradGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  workspace_size_list_.clear();
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }

  in_shapes_ = inputs[0]->GetShapeVector();
  param_->input_batch_ = LongToInt(in_shapes_[kDim0]);
  param_->input_channel_ = LongToInt(in_shapes_[kDim1]);
  param_->input_h_ = LongToInt(in_shapes_[height_index_]);
  param_->input_w_ = LongToInt(in_shapes_[width_index_]);

  out_shapes_ = inputs[1]->GetShapeVector();
  param_->output_batch_ = LongToInt(out_shapes_[kDim0]);
  param_->output_channel_ = LongToInt(out_shapes_[kDim1]);
  param_->output_h_ = LongToInt(out_shapes_[height_index_]);
  param_->output_w_ = LongToInt(out_shapes_[width_index_]);
  output_elements_ = std::accumulate(out_shapes_.begin(), out_shapes_.end(), 1, std::multiplies<size_t>());

  if (dim_ == kMaxPool3DGradGradDim) {
    reinterpret_cast<Pooling3DParameter *>(param_)->input_d_ = LongToInt(in_shapes_[depth_index_]);
    reinterpret_cast<Pooling3DParameter *>(param_)->output_d_ = LongToInt(out_shapes_[depth_index_]);
  }
  input_batch_stride_ =
    LongToSize(std::accumulate(in_shapes_.begin() + 1, in_shapes_.end(), 1, std::multiplies<int64_t>()));
  output_batch_stride_ =
    LongToSize(std::accumulate(out_shapes_.begin() + 1, out_shapes_.end(), 1, std::multiplies<int64_t>()));

  CheckInputVaild();
  CalPad();
  return static_cast<int>(KRET_OK);
}

bool MaxPoolGradGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolGradGradOutputsNum, kernel_name_);
  auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto *grad_addr = reinterpret_cast<float *>(inputs[kGradIndex]->addr);
  auto *dx_addr = reinterpret_cast<float *>(outputs[0]->addr);

  auto task = [input_addr, grad_addr, dx_addr, this](size_t start, size_t end) {
    auto ret = static_cast<int>(NNACL_OK);
    if (dim_ == kMaxPool2DGradGradDim) {
      ret = MaxPoolGradGrad(input_addr, grad_addr, dx_addr, start, end, param_);
    } else if (dim_ == kMaxPool3DGradGradDim) {
      ret =
        MaxPool3DGradGrad(input_addr, grad_addr, dx_addr, start, end, reinterpret_cast<Pooling3DParameter *>(param_));
    }
    if (ret != static_cast<int>(NNACL_OK)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', call NNACL MaxPoolWithArgmax function failed. Error code: " << ret;
      return false;
    }
    return true;
  };
  ParallelLaunchAutoSearch(task, output_elements_, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<KernelAttr> MaxPoolGradGradCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolGradGrad, MaxPool2DGradGradCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPool3DGradGrad, MaxPool3DGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
