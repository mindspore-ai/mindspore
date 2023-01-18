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

#include "plugin/device/ascend/kernel/rts/tensor_copy_slices.h"
#include <memory>
#include <numeric>
#include <functional>
#include <string>
#include "abstract/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/device/kernel_runtime.h"
#include "utils/ms_context.h"

using mindspore::ge::model_runner::MemcpyAsyncTaskInfo;
namespace mindspore {
namespace kernel {
constexpr auto kTensorCopySlicesInputSize = 2;
TensorCopySlices::TensorCopySlices() {}

TensorCopySlices::~TensorCopySlices() {}

bool TensorCopySlices::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (inputs.size() != kTensorCopySlicesInputSize) {
    MS_LOG(ERROR) << "inputs size is not 2";
    return false;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "outputs size is not 1";
    return false;
  }
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(ERROR) << "TensorCopySlices destMax > src size";
    return false;
  }

  auto status = aclrtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed!";
    return false;
  }
  status = aclrtMemcpyAsync(VoidPointerOffset(outputs[0]->addr, offset_), copy_size_, inputs[1]->addr, copy_size_,
                            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed!";
    return false;
  }
  return true;
}

bool TensorCopySlices::Init(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GetInputOutputInfo(anf_node);
  GetInputOutputTotalCount(anf_node);

  auto begin = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(anf_node, kAttrBegin);
  auto end = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(anf_node, kAttrEnd);
  auto strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(anf_node, kAttrStrides);

  CheckSliceValid(begin, end, strides, input_shape_);
  auto dim_offset = CalDimOffset(input_shape_);
  offset_ = CalOffset(begin, end, dim_offset) * abstract::TypeIdSize(input_type_id_);
  copy_size_ = GetCopySize(dim_offset, begin, end) * abstract::TypeIdSize(input_type_id_);
  return true;
}

void TensorCopySlices::GetInputOutputInfo(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_size = common::AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != kTensorCopySlicesInputSize) {
    MS_LOG(EXCEPTION) << "TensorCopySlices input size is not 2, got " << input_size;
  }
  input_type_id_ = AnfAlgo::GetPrevNodeOutputDeviceDataType(anf_node, 0);
  update_type_id_ = AnfAlgo::GetPrevNodeOutputDeviceDataType(anf_node, 0);
  output_type_id_ = AnfAlgo::GetOutputDeviceDataType(anf_node, 0);
  if (input_type_id_ != output_type_id_ || input_type_id_ != update_type_id_) {
    MS_LOG(EXCEPTION) << "Input and output of TensorCopySlices is not same, input type:" << input_type_id_
                      << " output_type_id_:" << output_type_id_;
  }

  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, 0);
  update_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(anf_node, 0);
}

void *TensorCopySlices::VoidPointerOffset(void *ptr, size_t offset) const {
  return static_cast<uint8_t *>(ptr) + offset;
}

void TensorCopySlices::GetInputOutputTotalCount(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_size = common::AnfAlgo::GetInputTensorNum(anf_node);
  if (input_size != kTensorCopySlicesInputSize) {
    MS_LOG(EXCEPTION) << "TensorCopySlices input size is not 2";
  }

  auto input_shape = AnfAlgo::GetInputDeviceShape(anf_node, 0);
  size_t total_size =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<size_t>(1), std::multiplies<>());
  total_size *= abstract::TypeIdSize(input_type_id_);
  MS_LOG(INFO) << "TensorCopySlices size[" << total_size << "]";
  // Shape and DType of input0 and output0 are same.
  mutable_input_size_list_.emplace_back(total_size);
  mutable_output_size_list_.emplace_back(total_size);

  auto update_shape = AnfAlgo::GetInputDeviceShape(anf_node, 1);
  size_t update_size =
    std::accumulate(update_shape.begin(), update_shape.end(), static_cast<size_t>(1), std::multiplies<>());
  update_size *= abstract::TypeIdSize(update_type_id_);
  mutable_input_size_list_.emplace_back(update_size);
}

std::vector<TaskInfoPtr> TensorCopySlices::GenTask(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (inputs.size() != kTensorCopySlicesInputSize) {
    MS_LOG(EXCEPTION) << "inputs size is not 2.";
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "outputs size is not 1.";
  }
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(EXCEPTION) << "TensorCopySlices input size " << inputs[0]->size << " is not equal to output size "
                      << outputs[0]->size;
  }

  stream_id_ = stream_id;
  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr1 =
    std::make_shared<MemcpyAsyncTaskInfo>(unique_name_, stream_id, outputs[0]->addr, outputs[0]->size, inputs[0]->addr,
                                          inputs[0]->size, ACL_MEMCPY_DEVICE_TO_DEVICE, NeedDump());
  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr2 = std::make_shared<MemcpyAsyncTaskInfo>(
    unique_name_, stream_id, VoidPointerOffset(outputs[0]->addr, offset_), copy_size_, inputs[1]->addr, copy_size_,
    ACL_MEMCPY_DEVICE_TO_DEVICE, NeedDump());
  return {task_info_ptr1, task_info_ptr2};
}

const std::vector<TypeId> data_type_list = {
  kNumberTypeInt,   kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
  kNumberTypeUInt,  kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,
  kNumberTypeFloat, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBool};
// If input's format is 5D, we will insert TransData before TensorCopySlices.
const std::vector<std::string> format_list = {kOpFormat_DEFAULT, kOpFormat_NCHW, kOpFormat_NHWC};

TensorCopySlicesDesc::TensorCopySlicesDesc() {}

TensorCopySlicesDesc::~TensorCopySlicesDesc() {}

// TensorCopySlices Register
std::vector<std::shared_ptr<kernel::KernelBuildInfo>> TensorCopySlicesDesc::GetKernelInfo(const CNodePtr &) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> tensor_copy_slices_build_info{};
  for (const auto &format : format_list) {
    for (const auto &type : data_type_list) {
      auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
      vector<string> input_format{format, format};
      vector<TypeId> input_type{type, type};
      vector<string> output_format{format};
      vector<TypeId> output_type{type};
      builder.SetInputsFormat(input_format);
      builder.SetInputsDeviceType(input_type);
      builder.SetOutputsFormat(output_format);
      builder.SetOutputsDeviceType(output_type);
      builder.SetProcessor(AICORE);
      builder.SetKernelType(RT_KERNEL);
      builder.SetFusionType(kPatternOpaque);
      builder.SetInputsKernelObjectType({KernelObjectType::TENSOR, KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({KernelObjectType::TENSOR});
      tensor_copy_slices_build_info.emplace_back(builder.Build());
    }
  }
  return tensor_copy_slices_build_info;
}
}  // namespace kernel
}  // namespace mindspore
