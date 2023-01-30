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

#include "plugin/device/cpu/kernel/mkldnn/batch_norm_grad_cpu_kernel.h"
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "mindspore/core/ops/grad/batch_norm_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormGradInputsNum = 6;
constexpr size_t kBatchNormGradOutputsNum = 3;
constexpr size_t kBatchNormGradInputShapeMaxSize = 4;
constexpr size_t kBatchNormGradInputShapeMinSize = 2;
constexpr size_t kScaleShiftNum = 2;
}  // namespace
bool BatchNormGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  base_operator_ = base_operator;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BatchNormGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BatchNormGrad ops failed!";
    return false;
  }
  is_train_ = kernel_ptr->get_is_training();
  epsilon_ = kernel_ptr->get_epsilon();
  kernel_name_ = kernel_ptr->GetPrim()->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  bool is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  return true;
}

int BatchNormGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }

  auto x_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  const size_t x_shape_size = x_shape.size();
  (void)x_shape.insert(x_shape.end(), kBatchNormGradInputShapeMaxSize - x_shape_size, 1);

  batch_size_ = x_shape[N];
  channel_ = x_shape[C];
  hw_size_ = x_shape[H] * x_shape[W];
  nhw_size_ = batch_size_ * hw_size_;

  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  dnnl::memory::desc scale_bias_desc = GetDefaultMemDesc(std::vector<int64_t>{kScaleShiftNum, channel_});
  auto prop_kind = dnnl::prop_kind::forward_inference;
  auto normalization_flags = dnnl::normalization_flags::use_scale_shift | dnnl::normalization_flags::use_global_stats;
  if (is_train_) {
    prop_kind = dnnl::prop_kind::forward_training;
    normalization_flags = dnnl::normalization_flags::use_scale_shift;
  }

  // fused Batch Normalization forward description
  auto desc = CreateDesc<dnnl::batch_normalization_forward::desc>(prop_kind, x_desc, epsilon_, normalization_flags);
  auto forward_prim_desc = CreateDesc<dnnl::batch_normalization_forward::primitive_desc>(desc, engine_);

  // fused Batch Normalization backward description
  auto backward_desc = CreateDesc<dnnl::batch_normalization_backward::desc>(dnnl::prop_kind::backward, x_desc, x_desc,
                                                                            epsilon_, normalization_flags);
  auto backward_prim_desc =
    CreateDesc<dnnl::batch_normalization_backward::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  auto wksp_desc = GetWorkspaceDesc(forward_prim_desc);
  auto mean = GetMeanDesc(forward_prim_desc);
  auto variance = GetVarianceDesc(forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::batch_normalization_backward>(backward_prim_desc);

  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, mean);
  AddArgument(DNNL_ARG_VARIANCE, variance);
  AddArgument(DNNL_ARG_SCALE_SHIFT, scale_bias_desc);
  AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
  AddArgument(DNNL_ARG_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, x_desc);
  AddArgument(DNNL_ARG_DIFF_SCALE_SHIFT, scale_bias_desc);

  InitWorkspaceSize(inputs);
  inputs_ = inputs;
  outputs_ = outputs;
  inputs_on_host_ = inputsOnHost;
  return KRET_OK;
}

void BatchNormGradCpuKernelMod::InitWorkspaceSize(const std::vector<KernelTensorPtr> &inputs) {
  size_t type_size = sizeof(float);
  auto shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  size_t tensor_size = static_cast<size_t>(shape[C]) * kScaleShiftNum * type_size;
  input_size_list_.pop_back();
  // [2, c] to store scale and bias
  (void)workspace_size_list_.emplace_back(tensor_size);
  // [2, c] to store diff_scale and diff_bias
  (void)workspace_size_list_.emplace_back(tensor_size);
}

bool BatchNormGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBatchNormGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBatchNormGradOutputsNum, kernel_name_);
  // From CPUKernelExecutor::LaunchKernel
  if (!Init(base_operator_, inputs_, outputs_)) {
    MS_LOG(ERROR) << "Re-init BatchNormGradCpuKernelMod while launching failed";
    return false;
  }
  auto resize_ret = Resize(base_operator_, inputs_, outputs_, inputs_on_host_);
  if (resize_ret != KRET_OK) {
    MS_LOG(ERROR) << "Resize BatchNormGradCpuKernelMod while launching failed: " << resize_ret;
    return false;
  }

  auto wksp_in = reinterpret_cast<float *>(workspace[SCALE_BIAS]->addr);
  auto scale_ret = memcpy_s(wksp_in, workspace[SCALE_BIAS]->size, inputs[SCALE]->addr, inputs[SCALE]->size);
  if (scale_ret != EOK) {
    MS_LOG(EXCEPTION) << "Scale memcpy error!";
  }
  auto max_size = workspace[SCALE_BIAS]->size - inputs[SCALE]->size;
  auto bias_ret = memset_s(wksp_in + (inputs[SCALE]->size / sizeof(float)), max_size, 0, max_size);
  if (bias_ret != EOK) {
    MS_LOG(EXCEPTION) << "Bias memset 0 error.";
  }

  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[Y_BACKPROP]->addr);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[X]->addr);
  SetArgumentHandle(DNNL_ARG_MEAN, inputs[SAVE_MEAN]->addr);
  SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[SAVE_VARIANCE]->addr);
  SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[SCALE_BIAS]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[DX]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SCALE_SHIFT, workspace[DIFF_SCALE_BIAS]->addr);
  ExecutePrimitive();

  auto wksp_out = reinterpret_cast<float *>(workspace[DIFF_SCALE_BIAS]->addr);
  auto diff_scale_ret = memcpy_s(outputs[DSCALE]->addr, outputs[DSCALE]->size, wksp_out, inputs[SCALE]->size);
  if (diff_scale_ret != EOK) {
    MS_LOG(EXCEPTION) << "Diff_scale memcpy to output[1] error.";
  }
  auto diff_bias_ret = memcpy_s(outputs[DBIAS]->addr, outputs[DBIAS]->size,
                                wksp_out + (outputs[DSCALE]->size / sizeof(float)), outputs[DBIAS]->size);
  if (diff_bias_ret != EOK) {
    MS_LOG(EXCEPTION) << "Diff_bias memcpy to  to output[2] error.";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNormGrad, BatchNormGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
