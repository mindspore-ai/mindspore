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
#include "nnacl/fp32/maxpool_with_argmax.h"
#include "nnacl/base/gather_base.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolingGradGradInputsNum = 3;
constexpr size_t kMaxPoolingGradGradOutputsNum = 1;
constexpr size_t kMaxPoolingGradGradWorkSpaceNum = 2;
constexpr size_t kGradIndex = 2;

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
  if (inputs.size() != kMaxPoolingGradGradInputsNum || outputs.size() != kMaxPoolingGradGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolingGradGradInputsNum << " and "
                  << kMaxPoolingGradGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
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

  param_ = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  MS_ERROR_IF_NULL(param_);
  param_->window_h_ = LongToInt(kernels_[kDim2]);
  param_->window_w_ = LongToInt(kernels_[kDim3]);
  param_->stride_h_ = LongToInt(strides_[kDim2]);
  param_->stride_w_ = LongToInt(strides_[kDim3]);
  return true;
}

void MaxPoolGradGradCpuKernelMod::CheckInputVaild() {
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
  MS_EXCEPTION_IF_ZERO("stride height", param_->stride_h_);
  MS_EXCEPTION_IF_ZERO("stride width", param_->stride_w_);
  if (pad_mode_ == PadMode::SAME) {
    int tmp_height = (param_->input_h_ / param_->stride_h_) * param_->stride_h_ == param_->input_h_
                       ? (param_->input_h_ / param_->stride_h_)
                       : (param_->input_h_ / param_->stride_h_) + 1;
    int pad_h = std::max<int>(0, (tmp_height - 1) * param_->stride_h_ + param_->window_h_ - param_->input_h_);
    param_->pad_u_ = pad_h / 2;

    int tmp_width = (param_->input_w_ / param_->stride_w_) * param_->stride_w_ == param_->input_w_
                      ? (param_->input_w_ / param_->stride_w_)
                      : (param_->input_w_ / param_->stride_w_) + 1;
    int pad_w = std::max<int>(0, (tmp_width - 1) * param_->stride_w_ + param_->window_w_ - param_->input_w_);
    param_->pad_l_ = pad_w / 2;
  }
}

void MaxPoolGradGradCpuKernelMod::InitWorkspace() {
  workspace_size_list_.push_back(input_size_list_[1]);
  auto index_size = std::accumulate(out_shapes_.begin(), out_shapes_.end(), sizeof(int32_t), std::multiplies<size_t>());
  workspace_size_list_.push_back(index_size);
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
  param_->input_h_ = LongToInt(in_shapes_[kDim2]);
  param_->input_w_ = LongToInt(in_shapes_[kDim3]);

  out_shapes_ = inputs[1]->GetShapeVector();
  param_->output_batch_ = LongToInt(out_shapes_[kDim0]);
  param_->output_channel_ = LongToInt(out_shapes_[kDim1]);
  param_->output_h_ = LongToInt(out_shapes_[kDim2]);
  param_->output_w_ = LongToInt(out_shapes_[kDim3]);

  CheckInputVaild();
  CalPad();
  InitWorkspace();
  return KRET_OK;
}

bool MaxPoolGradGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolingGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolingGradGradOutputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kMaxPoolingGradGradWorkSpaceNum, kernel_name_);
  auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<float *>(workspace[0]->addr);
  auto *index_addr = reinterpret_cast<int32_t *>(workspace[1]->addr);

  auto task = [input_addr, output_addr, index_addr, this](size_t start, size_t end) {
    auto ret = MaxPoolWithArgmax(input_addr, output_addr, index_addr, start, end, param_);
    if (ret != static_cast<int>(NNACL_OK)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', call NNACL MaxPoolWithArgmax function failed. Error code: " << ret;
    }
  };
  auto out_elements = param_->output_batch_ * param_->output_channel_ * param_->output_h_ * param_->output_w_;
  ParallelLaunchAutoSearch(task, out_elements, this, &parallel_search_info_, pool_);

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t indices_element_size = IntToLong(param_->output_channel_ * param_->output_h_ * param_->output_w_);
  int64_t limit = IntToLong(param_->input_channel_ * param_->input_h_ * param_->input_w_);
  size_t byte_inner_size = inner_size * sizeof(float);
  size_t byte_out_stride = indices_element_size * byte_inner_size;

  for (int b = 0; b < param_->input_batch_; b++) {
    auto *index_t = index_addr + b * param_->output_channel_ * param_->output_h_ * param_->output_w_;
    auto *grad_t = reinterpret_cast<float *>(inputs[kGradIndex]->addr) +
                   b * param_->input_channel_ * param_->input_h_ * param_->input_w_;
    auto *dx_t =
      reinterpret_cast<float *>(outputs[0]->addr) + b * param_->output_channel_ * param_->output_h_ * param_->output_w_;
    int ret = Gather(grad_t, outer_size, byte_inner_size, limit, index_t, indices_element_size, dx_t, byte_out_stride);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', error_code[" << ret << "]";
    }
  }
  return true;
}

std::vector<KernelAttr> MaxPoolGradGradCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolGradGrad, MaxPoolGradGradCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPool3DGradGrad, MaxPoolGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
