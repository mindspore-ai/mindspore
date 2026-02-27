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

#include "kernel/cpu/mkldnn/batch_norm_grad_cpu_kernel.h"
#include <map>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormGradInputShapeMaxSize = 4;
constexpr size_t kBatchNormGradInputShapeMinSize = 2;
}  // namespace
bool BatchNormGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  bool is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  return true;
}

int BatchNormGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }

  is_train_ = inputs[kIndex6]->GetValueWithCheck<bool>();
  epsilon_ = inputs[kIndex7]->GetValueWithCheck<float>();

  auto x_shape = inputs[kIndex0]->GetDeviceShapeVector();
  const size_t x_shape_size = x_shape.size();
  (void)x_shape.insert(x_shape.end(), kBatchNormGradInputShapeMaxSize - x_shape_size, 1);
  batch_size_ = x_shape[N];
  channel_ = x_shape[C];
  hw_size_ = x_shape[H] * x_shape[W];
  nhw_size_ = batch_size_ * hw_size_;

  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  dnnl::memory::desc scale_desc = GetDefaultMemDesc(std::vector<int64_t>{channel_});
  dnnl::memory::desc shift_desc = GetDefaultMemDesc(std::vector<int64_t>{channel_});
  auto prop_kind = dnnl::prop_kind::forward_inference;
  auto normalization_flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift |
                             dnnl::normalization_flags::use_global_stats;
  if (is_train_) {
    prop_kind = dnnl::prop_kind::forward_training;
    normalization_flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
  }

  // fused Batch Normalization forward description
  auto forward_prim_desc = CreateDesc<dnnl::batch_normalization_forward::primitive_desc>(
    engine_, prop_kind, x_desc, x_desc, epsilon_, normalization_flags);

  // fused Batch Normalization backward description
  auto backward_prim_desc = CreateDesc<dnnl::batch_normalization_backward::primitive_desc>(
    engine_, dnnl::prop_kind::backward, x_desc, x_desc, x_desc, epsilon_, normalization_flags, forward_prim_desc);
  auto wksp_desc = GetWorkspaceDesc(forward_prim_desc);
  auto mean = GetMeanDesc(forward_prim_desc);
  auto variance = GetVarianceDesc(forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::batch_normalization_backward>(backward_prim_desc);

  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, mean);
  AddArgument(DNNL_ARG_VARIANCE, variance);
  AddArgument(DNNL_ARG_SCALE, scale_desc);
  AddArgument(DNNL_ARG_DIFF_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, x_desc);
  AddArgument(DNNL_ARG_DIFF_SCALE, scale_desc);
  AddArgument(DNNL_ARG_DIFF_SHIFT, shift_desc);
  // Workspace is required only when forward was created with fused ReLU/add+ReLU flags.
  if (wksp_desc.get_ndims() != 0) {
    AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
  }

  InitWorkspaceSize(inputs);
  return KRET_OK;
}

void BatchNormGradCpuKernelMod::InitWorkspaceSize(const std::vector<KernelTensor *> &inputs) {
  size_t type_size = sizeof(float);
  auto shape = inputs[X]->GetDeviceShapeVector();
  // [c] for SCALE, [c] for DIFF_SCALE, [c] for DIFF_SHIFT
  size_t tensor_size = static_cast<size_t>(shape[C]) * type_size;
  workspace_size_list_.clear();
  (void)workspace_size_list_.emplace_back(tensor_size);  // SCALE_WKSP
  (void)workspace_size_list_.emplace_back(tensor_size);  // DIFF_SCALE_WKSP
  (void)workspace_size_list_.emplace_back(tensor_size);  // DIFF_SHIFT_WKSP
}

bool BatchNormGradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  // Copy SCALE into workspace to avoid any unexpected aliasing and to keep handles stable.
  auto scale_wksp = GetDeviceAddress<float>(workspace, SCALE_WKSP);
  auto scale_ret =
    memcpy_s(scale_wksp, workspace[SCALE_WKSP]->size(), inputs[SCALE]->device_ptr(), inputs[SCALE]->size());
  if (scale_ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
  }

  auto diff_scale_wksp = GetDeviceAddress<float>(workspace, DIFF_SCALE_WKSP);
  auto diff_shift_wksp = GetDeviceAddress<float>(workspace, DIFF_SHIFT_WKSP);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[Y_BACKPROP]->device_ptr());
  SetArgumentHandle(DNNL_ARG_SRC, inputs[X]->device_ptr());
  SetArgumentHandle(DNNL_ARG_MEAN, inputs[SAVE_MEAN]->device_ptr());
  SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[SAVE_VARIANCE]->device_ptr());
  SetArgumentHandle(DNNL_ARG_SCALE, workspace[SCALE_WKSP]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[DX]->device_ptr());
  // Use workspace as intermediate buffer for diff_scale/diff_shift, then copy back to outputs.
  SetArgumentHandle(DNNL_ARG_DIFF_SCALE, diff_scale_wksp);
  SetArgumentHandle(DNNL_ARG_DIFF_SHIFT, diff_shift_wksp);
  // If forward reserved workspace is required (fused ReLU/add+ReLU), it is passed via RESERVE input.
  // Only set the handle when the argument was added in Resize().
  if (arguments_.find(DNNL_ARG_WORKSPACE) != arguments_.end()) {
    SetArgumentHandle(DNNL_ARG_WORKSPACE, inputs[RESERVE]->device_ptr());
  }
  ExecutePrimitive();

  auto diff_scale_ret =
    memcpy_s(outputs[DSCALE]->device_ptr(), outputs[DSCALE]->size(), diff_scale_wksp, sizeof(float) * channel_);
  auto diff_shift_ret =
    memcpy_s(outputs[DBIAS]->device_ptr(), outputs[DBIAS]->size(), diff_shift_wksp, sizeof(float) * channel_);
  if (diff_scale_ret != EOK || diff_shift_ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
  }
  return true;
}

#define BATCH_NORM_GRAD_CPU_REG(MS)                      \
  KernelAttr()                                           \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(MS)                                    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32) \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)   \
    .AddOutputAttr(MS)                                   \
    .AddOutputAttr(MS)                                   \
    .AddOutputAttr(MS)

std::vector<KernelAttr> BatchNormGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {BATCH_NORM_GRAD_CPU_REG(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNormGrad, BatchNormGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
