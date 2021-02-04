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
#include <string>
#include "backend/kernel_compiler/cpu/mkldnn/batch_norm_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void BatchNormCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t tensor_size = shape[1] * 2 * type_size;  // [2, c] to store scale and bias
  workspace_size_list_.emplace_back(tensor_size);
}

void BatchNormCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  is_train = AnfAlgo::GetNodeAttr<bool>(kernel_node, "is_training");
  momentum = AnfAlgo::GetNodeAttr<float>(kernel_node, "momentum");
  std::vector<size_t> x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (x_shape.size() != 4) {
    MS_LOG(EXCEPTION) << "Batchnorm only support nchw input!";
  }
  batch_size = x_shape[0];
  channel = x_shape[1];
  hw_size = x_shape[2] * x_shape[3];
  nhw_size = x_shape[0] * hw_size;
  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  dnnl::memory::desc scale_bias_desc = GetDefaultMemDesc({2, channel});
  auto epsilon = AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  auto prop_kind = dnnl::prop_kind::forward_inference;
  auto normalization_flags = dnnl::normalization_flags::use_scale_shift | dnnl::normalization_flags::use_global_stats;
  if (is_train) {
    prop_kind = dnnl::prop_kind::forward_training;
    normalization_flags = dnnl::normalization_flags::use_scale_shift;
  }
  dnnl::batch_normalization_forward::desc desc =
    dnnl::batch_normalization_forward::desc(prop_kind, x_desc, epsilon, normalization_flags);
  auto prim_desc = dnnl::batch_normalization_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::batch_normalization_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, prim_desc.mean_desc());
  AddArgument(DNNL_ARG_VARIANCE, prim_desc.variance_desc());
  AddArgument(DNNL_ARG_SCALE_SHIFT, scale_bias_desc);
  AddArgument(DNNL_ARG_WORKSPACE, prim_desc.workspace_desc());
  AddArgument(DNNL_ARG_DST, x_desc);
}

bool BatchNormCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 5 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Error input output size!";
  }
  auto wksp = reinterpret_cast<float *>(workspace[0]->addr);
  auto scale_ret = memcpy_s(wksp, workspace[0]->size, inputs[1]->addr, inputs[1]->size);
  auto max_size = workspace[0]->size - inputs[1]->size;
  auto bias_ret = memcpy_s(wksp + (inputs[1]->size / sizeof(float)), max_size, inputs[2]->addr, inputs[2]->size);
  if (scale_ret != 0 || bias_ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
    return false;
  }
  if (is_train) {
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
      moving_mean[i] = moving_mean[i] * (1 - momentum) + mean[i] * momentum;
      moving_variance[i] = moving_variance[i] * (1 - momentum) + variance[i] * momentum;
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
}  // namespace kernel
}  // namespace mindspore
