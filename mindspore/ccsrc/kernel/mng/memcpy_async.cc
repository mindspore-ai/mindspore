/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/mng/memcpy_async.h"

#include <memory>
#include <string>

#include "runtime/mem.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "common/trans.h"

using ge::model_runner::MemcpyAsyncTaskInfo;
using MemcpyAsyncTaskInfoPtr = std::shared_ptr<MemcpyAsyncTaskInfo>;

namespace mindspore {
namespace kernel {
MemCpyAsyncKernel::MemCpyAsyncKernel() {}

MemCpyAsyncKernel::~MemCpyAsyncKernel() {}

bool MemCpyAsyncKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                               const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  auto stream = reinterpret_cast<rtStream_t>(stream_ptr);

  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "inputs size is not one";
    return false;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "outputs size is not one";
    return false;
  }

  if (inputs[0]->addr == outputs[0]->addr) {
    MS_LOG(INFO) << "input addr is same with output addr , no need exe memcpy async";
    return true;
  }
  rtError_t status = rtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                   RT_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op rtMemcpyAsync failed!";
    return false;
  }
  return true;
}

bool MemCpyAsyncKernel::Init(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GetInputOutputDataType(anf_node);
  GetInputOutputTotalCount(anf_node);
  return true;
}

void MemCpyAsyncKernel::GetInputOutputDataType(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_size = AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync input size is not 1";
  }
  input_type_id_ = AnfAlgo::GetPrevNodeOutputInferDataType(anf_node, 0);
}

void MemCpyAsyncKernel::GetInputOutputTotalCount(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_size = AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync input size is not 1";
  }
  size_t type_size = trans::TypeIdSize(input_type_id_);
  std::vector<size_t> shape_i = AnfAlgo::GetInputDeviceShape(anf_node, 0);
  size_t total_size = 1;
  for (size_t i = 0; i < shape_i.size(); i++) {
    total_size = total_size * shape_i[i];
  }
  total_size *= type_size;
  MS_LOG(INFO) << "MemCpyAsync size[" << total_size << "]";
  input_size_list_.emplace_back(total_size);
  output_size_list_.emplace_back(total_size);
}

std::vector<TaskInfoPtr> MemCpyAsyncKernel::GenTask(const vector<mindspore::kernel::AddressPtr> &inputs,
                                                    const vector<mindspore::kernel::AddressPtr> & /*workspace*/,
                                                    const vector<mindspore::kernel::AddressPtr> &outputs,
                                                    uint32_t stream_id) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync op inputs is not one";
  }

  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync op output is not one";
  }

  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr = std::make_shared<MemcpyAsyncTaskInfo>(
    stream_id, outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size, RT_MEMCPY_DEVICE_TO_DEVICE);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}

const std::vector<TypeId> data_type_list{kNumberTypeInt,     kNumberTypeInt8,   kNumberTypeInt16, kNumberTypeInt32,
                                         kNumberTypeInt64,   kNumberTypeUInt,   kNumberTypeUInt8, kNumberTypeUInt16,
                                         kNumberTypeUInt32,  kNumberTypeUInt64, kNumberTypeFloat, kNumberTypeFloat16,
                                         kNumberTypeFloat32, kNumberTypeFloat64};
const std::vector<std::string> format_list = {kOpFormat_DEFAULT,  kOpFormat_NCHW,   kOpFormat_NHWC,
                                              kOpFormat_NC1HWC0,  kOpFormat_FRAC_Z, kOpFormat_NC1KHKWHWC0,
                                              kOpFormat_C1HWNCoC0};

MemCpyAsyncDesc::MemCpyAsyncDesc() {}

MemCpyAsyncDesc::~MemCpyAsyncDesc() {}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> MemCpyAsyncDesc::GetKernelInfo() {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> memcpy_build_info{};
  for (const auto &format : format_list) {
    for (const auto &type : data_type_list) {
      auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
      vector<string> input_format{format};
      vector<TypeId> input_type{type};
      vector<string> output_format{format};
      vector<TypeId> output_type{type};
      builder.SetInputsFormat(input_format);
      builder.SetInputsDeviceType(input_type);
      builder.SetOutputsFormat(output_format);
      builder.SetOutputsDeviceType(output_type);
      builder.SetProcessor(AICORE);
      builder.SetKernelType(RT_KERNEL);
      builder.SetFusionType(OPAQUE);
      memcpy_build_info.emplace_back(builder.Build());
    }
  }
  return memcpy_build_info;
}
}  // namespace kernel
}  // namespace mindspore
