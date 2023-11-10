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

#include "plugin/device/cpu/kernel/mkldnn/batch_norm_cpu_kernel.h"
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormInputShapeMaxSize = 4;
}  // namespace

bool BatchNormCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  bool is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  return true;
}

int BatchNormCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }

  is_train_ = inputs[kIndex5]->GetValueWithCheck<bool>();
  epsilon_ = inputs[kIndex6]->GetValueWithCheck<float>();
  momentum_ = inputs[kIndex7]->GetValueWithCheck<float>();

  auto x_shape = inputs[kIndex0]->GetDeviceShapeVector();
  (void)x_shape.insert(x_shape.end(), kBatchNormInputShapeMaxSize - x_shape.size(), 1);

  batch_size_ = x_shape[0];
  channel_ = x_shape[kIndex1];
  hw_size_ = x_shape[kIndex2] * x_shape[kIndex3];
  nhw_size_ = x_shape[0] * hw_size_;

  auto prop_kind = dnnl::prop_kind::forward_inference;
  auto normalization_flags = dnnl::normalization_flags::use_scale_shift | dnnl::normalization_flags::use_global_stats;
  if (is_train_) {
    prop_kind = dnnl::prop_kind::forward_training;
    normalization_flags = dnnl::normalization_flags::use_scale_shift;
  }

  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  auto scale_bias_shape = std::vector<int64_t>{2, channel_};
  dnnl::memory::desc scale_bias_desc = GetDefaultMemDesc(scale_bias_shape);
  auto desc = CreateDesc<dnnl::batch_normalization_forward::desc>(prop_kind, x_desc, epsilon_, normalization_flags);
  auto prim_desc = CreateDesc<dnnl::batch_normalization_forward::primitive_desc>(desc, engine_);
  auto wksp_desc = GetWorkspaceDesc(prim_desc);
  auto mean = GetMeanDesc(prim_desc);
  auto variance = GetVarianceDesc(prim_desc);
  primitive_ = CreatePrimitive<dnnl::batch_normalization_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, mean);
  AddArgument(DNNL_ARG_VARIANCE, variance);
  AddArgument(DNNL_ARG_SCALE_SHIFT, scale_bias_desc);
  AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
  AddArgument(DNNL_ARG_DST, x_desc);

  InitWorkspaceSize(inputs);
  return KRET_OK;
}

void BatchNormCpuKernelMod::InitWorkspaceSize(const std::vector<KernelTensor *> &inputs) {
  size_t type_size = sizeof(float);
  auto shape = inputs[0]->GetDeviceShapeVector();
  size_t tensor_size = static_cast<size_t>(shape[kIndex1]) * 2 * type_size;  // [2, c] to store scale and bias
  (void)workspace_size_list_.emplace_back(tensor_size);
}

bool BatchNormCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &workspace,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto wksp = reinterpret_cast<float *>(workspace[0]->device_ptr());
  auto scale_ret = memcpy_s(wksp, workspace[0]->size(), inputs[kIndex1]->device_ptr(), inputs[kIndex1]->size());
  auto max_size = workspace[0]->size() - inputs[kIndex1]->size();
  auto bias_ret = memcpy_s(wksp + (inputs[kIndex1]->size() / sizeof(float)), max_size, inputs[kIndex2]->device_ptr(),
                           inputs[kIndex2]->size());
  if (scale_ret != EOK || bias_ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
  }
  if (is_train_) {
    SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->device_ptr());
    SetArgumentHandle(DNNL_ARG_MEAN, outputs[kIndex3]->device_ptr());
    SetArgumentHandle(DNNL_ARG_VARIANCE, outputs[kIndex4]->device_ptr());
    SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[0]->device_ptr());
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->device_ptr());
    ExecutePrimitive();

    auto moving_mean = reinterpret_cast<float *>(inputs[kIndex3]->device_ptr());
    auto moving_variance = reinterpret_cast<float *>(inputs[kIndex4]->device_ptr());
    auto mean = reinterpret_cast<float *>(outputs[kIndex3]->device_ptr());
    auto variance = reinterpret_cast<float *>(outputs[kIndex4]->device_ptr());
    for (size_t i = 0; i < inputs[kIndex3]->size() / sizeof(float); ++i) {
      moving_mean[i] = moving_mean[i] * (1 - momentum_) + mean[i] * momentum_;
      moving_variance[i] = moving_variance[i] * (1 - momentum_) + variance[i] * momentum_;
    }
  } else {
    SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->device_ptr());
    SetArgumentHandle(DNNL_ARG_MEAN, inputs[kIndex3]->device_ptr());
    SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[kIndex4]->device_ptr());
    SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[0]->device_ptr());
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->device_ptr());
    ExecutePrimitive();
  }
  return true;
}

#define BATCH_NORM_CPU_REG(T)                            \
  KernelAttr()                                           \
    .AddInputAttr(T)                                     \
    .AddInputAttr(T)                                     \
    .AddInputAttr(T)                                     \
    .AddInputAttr(T)                                     \
    .AddInputAttr(T)                                     \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32) \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32) \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)   \
    .AddOutputAttr(T)                                    \
    .AddOutputAttr(T)                                    \
    .AddOutputAttr(T)                                    \
    .AddOutputAttr(T)                                    \
    .AddOutputAttr(T)

std::vector<KernelAttr> BatchNormCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {BATCH_NORM_CPU_REG(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNorm, BatchNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
