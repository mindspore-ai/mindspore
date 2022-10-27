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

#include "plugin/device/cpu/kernel/mkldnn/log_softmax_grad_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogSoftmaxGradInputsNum = 2;
constexpr size_t kLogSoftmaxGradOutputsNum = 1;
}  // namespace

bool LogSoftmaxGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSoftmaxGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSoftmaxGradOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int LogSoftmaxGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::LogSoftmaxGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  axis_ = kernel_ptr->get_axis();
  auto src_shape = inputs[0]->GetDeviceShapeAdaptively();
  if (axis_ >= SizeToLong(src_shape.size())) {
    axis_ = SizeToLong(src_shape.size()) - 1;
  }
  while (axis_ < 0) {
    axis_ += SizeToLong(src_shape.size());
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  auto desc = CreateDesc<dnnl::logsoftmax_forward::desc>(dnnl::prop_kind::forward_training, src_desc, axis_);
  auto prim_desc = CreateDesc<dnnl::logsoftmax_forward::primitive_desc>(desc, engine_);
  // backward description
  auto backward_desc = CreateDesc<dnnl::logsoftmax_backward::desc>(src_desc, src_desc, axis_);
  auto backward_prim_desc = CreateDesc<dnnl::logsoftmax_backward::primitive_desc>(backward_desc, engine_, prim_desc);
  primitive_ = CreatePrimitive<dnnl::logsoftmax_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, src_desc);
  return ret;
}

bool LogSoftmaxGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSoftmaxGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSoftmaxGradOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_DST, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogSoftmaxGrad, LogSoftmaxGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
