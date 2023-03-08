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

#include "plugin/device/ascend/kernel/rts/memcpy_async.h"
#include <memory>
#include <string>
#include "abstract/utils.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"

using mindspore::ge::model_runner::MemcpyAsyncTaskInfo;
using MemcpyAsyncTaskInfoPtr = std::shared_ptr<MemcpyAsyncTaskInfo>;

namespace mindspore {
namespace kernel {
MemCpyAsyncKernel::MemCpyAsyncKernel() {}

MemCpyAsyncKernel::~MemCpyAsyncKernel() {}

bool MemCpyAsyncKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "inputs size is not one";
    return false;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "outputs size is not one";
    return false;
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  if (inputs[0]->addr == outputs[0]->addr) {
    MS_LOG(INFO) << "input addr is same with output addr , no need exe memcpy async";
    return true;
  }
  if (outputs[0]->size < inputs[0]->size) {
    MS_LOG(EXCEPTION) << "aclrtMemcpyAsync destMax " << outputs[0]->size << " is less than src size "
                      << inputs[0]->size;
  }
  // input x -> memcpy_async -> AllReduce
  if (outputs[0]->size > inputs[0]->size) {
    MS_LOG(WARNING) << "aclrtMemcpyAsync destMax > src size";
  }
  rtError_t status = aclrtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed!";
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
  size_t input_size = common::AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync input size is not 1, got " << input_size;
  }
  input_type_id_ = AnfAlgo::GetPrevNodeOutputDeviceDataType(anf_node, 0);
}

void MemCpyAsyncKernel::GetInputOutputTotalCount(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_size = common::AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync input size is not 1, got " << input_size;
  }
  size_t type_size = abstract::TypeIdSize(input_type_id_);
  auto shape_i = AnfAlgo::GetInputDeviceShape(anf_node, 0);
  size_t total_size = 1;
  for (size_t i = 0; i < shape_i.size(); i++) {
    total_size = SizetMulWithOverflowCheck(total_size, static_cast<size_t>(shape_i[i]));
  }
  total_size = SizetMulWithOverflowCheck(total_size, type_size);
  MS_LOG(INFO) << "MemCpyAsync size[" << total_size << "]";
  mutable_input_size_list_.emplace_back(total_size);
  mutable_output_size_list_.emplace_back(total_size);
}

std::vector<TaskInfoPtr> MemCpyAsyncKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &,
                                                    const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync op inputs is not one";
  }

  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "MemCpyAsync op output is not one";
  }

  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[0]);
  if (outputs[0]->size < inputs[0]->size) {
    MS_LOG(EXCEPTION) << "aclrtMemcpyAsync destMax < src size";
  }
  // input x -> memcpy_async -> AllReduce
  if (outputs[0]->size > inputs[0]->size) {
    MS_LOG(WARNING) << "aclrtMemcpyAsync destMax > src size";
  }

  stream_id_ = stream_id;
  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr =
    std::make_shared<MemcpyAsyncTaskInfo>(unique_name_, stream_id, outputs[0]->addr, outputs[0]->size, inputs[0]->addr,
                                          inputs[0]->size, ACL_MEMCPY_DEVICE_TO_DEVICE, NeedDump());
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}

const std::vector<TypeId> data_type_list = {
  kNumberTypeInt,   kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
  kNumberTypeUInt,  kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,
  kNumberTypeFloat, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBool};
const std::vector<std::string> format_list = {kOpFormat_DEFAULT,  kOpFormat_NCHW,   kOpFormat_NHWC,
                                              kOpFormat_NC1HWC0,  kOpFormat_FRAC_Z, kOpFormat_NC1KHKWHWC0,
                                              kOpFormat_C1HWNCoC0};

MemCpyAsyncDesc::MemCpyAsyncDesc() {}

MemCpyAsyncDesc::~MemCpyAsyncDesc() {}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> MemCpyAsyncDesc::GetKernelInfo(const CNodePtr &) {
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
      builder.SetFusionType(kPatternOpaque);
      builder.SetInputsKernelObjectType({KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({KernelObjectType::TENSOR});
      memcpy_build_info.emplace_back(builder.Build());
    }
  }
  return memcpy_build_info;
}
}  // namespace kernel
}  // namespace mindspore
