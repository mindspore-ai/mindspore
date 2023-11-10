/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/softmax_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool SoftmaxCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return true;
}

int SoftmaxCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto src_shape = inputs[kIndex0]->GetShapeVector();
  axis_list_ = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (axis_list_.size() != 1) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", the parameter 'axis' only support int type on CPU, but got tuple.";
  }
  int axis = axis_list_[0];
  while (axis < 0) {
    axis += SizeToInt(src_shape.size());
  }

  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  auto desc = CreateDesc<dnnl::softmax_forward::desc>(dnnl::prop_kind::forward_training, src_desc, axis);
  auto prim_desc = CreateDesc<dnnl::softmax_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::softmax_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);

  return KRET_OK;
}

bool SoftmaxCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                 const std::vector<kernel::KernelTensor *> &,
                                 const std::vector<kernel::KernelTensor *> &outputs) {
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->device_ptr());
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Softmax, SoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
