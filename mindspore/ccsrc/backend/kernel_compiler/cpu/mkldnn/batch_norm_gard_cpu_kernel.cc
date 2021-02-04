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
#include "backend/kernel_compiler/cpu/mkldnn/batch_norm_gard_cpu_kernel.h"

#include <string>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void BatchNormGradCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t tensor_size = shape[1] * 2 * type_size;
  input_size_list_.pop_back();
  // [2, c] to store scale and bias
  workspace_size_list_.emplace_back(tensor_size);
  // [2, c] to store diff_scale and diff_bias
  workspace_size_list_.emplace_back(tensor_size);
}

void BatchNormGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (x_shape.size() != 4) {
    MS_LOG(EXCEPTION) << "Fused batchnorm only support nchw input!";
  }
  batch_size = x_shape[0];
  channel = x_shape[1];
  hw_size = x_shape[2] * x_shape[3];
  nhw_size = x_shape[0] * hw_size;
  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  dnnl::memory::desc scale_bias_desc = GetDefaultMemDesc({2, channel});
  auto epsilon = AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  auto prop_kind = dnnl::prop_kind::forward_training;
  auto normalization_flags = dnnl::normalization_flags::use_scale_shift;

  // fused batch normalization forward description
  dnnl::batch_normalization_forward::desc desc =
    dnnl::batch_normalization_forward::desc(prop_kind, x_desc, epsilon, normalization_flags);
  auto forward_prim_desc = dnnl::batch_normalization_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());

  // fused batch normalization backward description
  dnnl::batch_normalization_backward::desc backward_desc =
    dnnl::batch_normalization_backward::desc(dnnl::prop_kind::backward, x_desc, x_desc, epsilon, normalization_flags);
  auto backward_prim_desc = dnnl::batch_normalization_backward::primitive_desc(
    backward_desc, MKLKernelEngine::Get().engine(), forward_prim_desc);
  primitive_ = std::make_shared<dnnl::batch_normalization_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, forward_prim_desc.mean_desc());
  AddArgument(DNNL_ARG_VARIANCE, forward_prim_desc.variance_desc());
  AddArgument(DNNL_ARG_SCALE_SHIFT, scale_bias_desc);
  AddArgument(DNNL_ARG_WORKSPACE, forward_prim_desc.workspace_desc());
  AddArgument(DNNL_ARG_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, x_desc);
  AddArgument(DNNL_ARG_DIFF_SCALE_SHIFT, scale_bias_desc);
}

bool BatchNormGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 5 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Error input output size!";
  }
  auto wksp_in = reinterpret_cast<float *>(workspace[0]->addr);
  auto scale_ret = memcpy_s(wksp_in, workspace[0]->size, inputs[2]->addr, inputs[2]->size);
  auto max_size = workspace[0]->size - inputs[2]->size;
  auto bias_ret = memset_s(wksp_in + (inputs[2]->size / sizeof(float)), max_size, 0., max_size);
  if (scale_ret != 0 && bias_ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
    return false;
  }

  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_MEAN, inputs[3]->addr);
  SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[4]->addr);
  SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SCALE_SHIFT, workspace[1]->addr);
  ExecutePrimitive();

  auto wksp_out = reinterpret_cast<float *>(workspace[1]->addr);
  auto diff_scale_ret = memcpy_s(outputs[1]->addr, outputs[1]->size, wksp_out, inputs[2]->size);
  auto diff_bias_ret =
    memcpy_s(outputs[2]->addr, outputs[2]->size, wksp_out + (outputs[1]->size / sizeof(float)), outputs[2]->size);
  if (diff_scale_ret != 0 || diff_bias_ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy_s error.";
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
