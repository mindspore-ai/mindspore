/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/addn_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "utils/ms_utils.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAddNInputsMinNum = 2;
constexpr size_t kAddNOutputsNum = 1;

void AddInt(const int *in_0, const int *in_1, int *out, int start, int end) {
  int ret = ElementAddInt(in_0 + start, in_1 + start, out + start, end - start);
  if (ret != NNACL_OK) {
    MS_LOG(EXCEPTION) << "Add failed.";
  }
}

template <typename T>
void AddT(const T *in0, const T *in1, T *out, int start, int end) {
  for (int index = start; index < end; index++) {
    out[index] = in0[index] + in1[index];
  }
}
}  // namespace

template <typename T>
void AddNCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_num_ = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num_ < kAddNInputsMinNum) {
    MS_LOG(EXCEPTION) << "Input numbers should not less " << kAddNInputsMinNum << ", but got " << input_num_;
  }
  CheckParam(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  std::vector<size_t> src0_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src1_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  dnnl::memory::desc src0_mem_desc = GetDefaultMemDesc(src0_shape);
  dnnl::memory::desc src1_mem_desc = GetDefaultMemDesc(src1_shape);
  dnnl::memory::desc dst_mem_desc = GetDefaultMemDesc(dst_shape);
  auto desc = CreateDesc<dnnl::binary::desc>(dnnl::algorithm::binary_add, src0_mem_desc, src1_mem_desc, dst_mem_desc);
  auto prim_desc = CreateDesc<dnnl::binary::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::binary>(prim_desc);
  AddArgument(DNNL_ARG_SRC_0, src0_mem_desc);
  AddArgument(DNNL_ARG_SRC_1, src1_mem_desc);
  AddArgument(DNNL_ARG_DST, dst_mem_desc);
}

template <typename T>
bool AddNCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAddNOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    SetArgumentHandle(DNNL_ARG_SRC_0, inputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_SRC_1, inputs[1]->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
    ExecutePrimitive();
    for (size_t index = 2; index < input_num_; ++index) {
      SetArgumentHandle(DNNL_ARG_SRC_0, outputs[0]->addr);
      SetArgumentHandle(DNNL_ARG_SRC_1, inputs[index]->addr);
      SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
      ExecutePrimitive();
    }
  } else if (dtype_ == kNumberTypeInt32) {
    size_t elements_num = outputs[0]->size / sizeof(int);
    const auto input_0 = reinterpret_cast<int *>(inputs[0]->addr);
    const auto input_1 = reinterpret_cast<int *>(inputs[1]->addr);
    auto output = reinterpret_cast<int *>(outputs[0]->addr);
    auto task_0 = std::bind(AddInt, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);
    for (size_t index = 2; index < input_num_; ++index) {
      const auto input = reinterpret_cast<int *>(inputs[index]->addr);
      auto task = std::bind(AddInt, input, output, output, std::placeholders::_1, std::placeholders::_2);
      ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
    }
  } else {
    size_t elements_num = outputs[0]->size / sizeof(T);
    const auto input_0 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto input_1 = reinterpret_cast<T *>(inputs[1]->addr);
    auto output = reinterpret_cast<T *>(outputs[0]->addr);
    auto task_0 = std::bind(AddT<T>, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);
    for (size_t index = 2; index < input_num_; ++index) {
      const auto input = reinterpret_cast<T *>(inputs[index]->addr);
      auto task = std::bind(AddT<T>, input, output, output, std::placeholders::_1, std::placeholders::_2);
      ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
    }
  }
  return true;
}

template <typename T>
void AddNCpuKernelMod<T>::CheckParam(const CNodePtr &kernel_node) {
  auto src0_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (src0_shape != dst_shape) {
    MS_LOG(EXCEPTION) << "AddN output shape must be equal to input shape.";
  }
  for (size_t index = 1; index < input_num_; ++index) {
    auto src_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index);
    if (src0_shape != src_shape) {
      MS_LOG(EXCEPTION) << "AddN input shapes must be equal.";
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
