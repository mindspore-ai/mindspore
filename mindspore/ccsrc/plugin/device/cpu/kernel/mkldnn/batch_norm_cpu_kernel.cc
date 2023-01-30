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
#include "utils/ms_utils.h"
#include "mindspore/core/ops/batch_norm.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormInputsNum = 5;
constexpr size_t kBatchNormOutputsNum = 5;
constexpr size_t kBatchNormInputShapeMaxSize = 4;
}  // namespace

bool BatchNormCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BatchNorm>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BatchNorm ops failed!";
    return false;
  }

  kernel_name_ = kernel_ptr->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  bool is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }

  base_operator_ = base_operator;
  is_train_ = kernel_ptr->get_is_training();
  momentum_ = kernel_ptr->get_momentum();
  epsilon_ = kernel_ptr->get_epsilon();
  return true;
}

int BatchNormCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }

  auto x_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  (void)x_shape.insert(x_shape.end(), kBatchNormInputShapeMaxSize - x_shape.size(), 1);

  batch_size_ = x_shape[0];
  channel_ = x_shape[1];
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
  inputs_ = inputs;
  outputs_ = outputs;
  inputs_on_host_ = inputsOnHost;
  return KRET_OK;
}

void BatchNormCpuKernelMod::InitWorkspaceSize(const std::vector<KernelTensorPtr> &inputs) {
  size_t type_size = sizeof(float);
  auto shape = inputs[0]->GetDeviceShapeAdaptively();
  size_t tensor_size = static_cast<size_t>(shape[1]) * 2 * type_size;  // [2, c] to store scale and bias
  (void)workspace_size_list_.emplace_back(tensor_size);
}

bool BatchNormCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBatchNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBatchNormOutputsNum, kernel_name_);
  // From CPUKernelExecutor::LaunchKernel
  if (!Init(base_operator_, inputs_, outputs_)) {
    MS_LOG(ERROR) << "Re-init BatchNormCpuKernelMod while launching failed";
    return false;
  }
  auto resize_ret = Resize(base_operator_, inputs_, outputs_, inputs_on_host_);
  if (resize_ret != KRET_OK) {
    MS_LOG(ERROR) << "Resize BatchNormCpuKernelMod while launching failed: " << resize_ret;
    return false;
  }
  auto wksp = reinterpret_cast<float *>(workspace[0]->addr);
  auto scale_ret = memcpy_s(wksp, workspace[0]->size, inputs[1]->addr, inputs[1]->size);
  auto max_size = workspace[0]->size - inputs[1]->size;
  auto bias_ret = memcpy_s(wksp + (inputs[1]->size / sizeof(float)), max_size, inputs[2]->addr, inputs[2]->size);
  if (scale_ret != EOK || bias_ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
  }
  if (is_train_) {
    SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_MEAN, outputs[3]->addr);
    SetArgumentHandle(DNNL_ARG_VARIANCE, outputs[4]->addr);
    SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[0]->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
    ExecutePrimitive();

    auto moving_mean = reinterpret_cast<float *>(inputs[3]->addr);
    auto moving_variance = reinterpret_cast<float *>(inputs[4]->addr);
    auto mean = reinterpret_cast<float *>(outputs[3]->addr);
    auto variance = reinterpret_cast<float *>(outputs[4]->addr);
    for (size_t i = 0; i < inputs[3]->size / sizeof(float); ++i) {
      moving_mean[i] = moving_mean[i] * (1 - momentum_) + mean[i] * momentum_;
      moving_variance[i] = moving_variance[i] * (1 - momentum_) + variance[i] * momentum_;
    }
  } else {
    SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_MEAN, inputs[3]->addr);
    SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[4]->addr);
    SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[0]->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
    ExecutePrimitive();
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNorm, BatchNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
