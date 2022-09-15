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

#include "plugin/device/cpu/kernel/mkldnn/log_softmax_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "mindspore/core/ops/log_softmax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogSoftmaxInputsNum = 1;
constexpr size_t kLogSoftmaxOutputsNum = 1;
}  // namespace

bool LogSoftmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::make_shared<ops::LogSoftmax>(base_operator->GetPrim());
  axis_ori_ = LongToInt(kernel_ptr->get_axis());
  return true;
}

int LogSoftmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSoftmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSoftmaxOutputsNum, kernel_name_);

  const auto &src_shape = inputs.at(kIndex0)->GetShapeVector();
  axis_ = axis_ori_ < 0 ? (axis_ori_ + SizeToInt(src_shape.size())) : axis_ori_;

  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  auto desc = CreateDesc<dnnl::logsoftmax_forward::desc>(dnnl::prop_kind::forward_inference, src_desc, axis_);
  auto prim_desc = CreateDesc<dnnl::logsoftmax_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::logsoftmax_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
  return KRET_OK;
}

bool LogSoftmaxCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();

  // Filter positive values
  auto output_ptr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t num = outputs[0]->size / sizeof(float);

  auto task = [output_ptr](size_t start_index, size_t end_index) {
    for (size_t i = start_index; i < end_index; i++) {
      if (output_ptr[i] > 0) {
        output_ptr[i] = 0;
      }
    }
  };

  ParallelLaunchAutoSearch(task, num, this, &parallel_search_info_);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogSoftmax, LogSoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
