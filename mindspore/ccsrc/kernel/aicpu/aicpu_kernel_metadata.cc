/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "kernel/aicpu/aicpu_kernel_metadata.h"
#include <memory>
#include <string>
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
constexpr auto kInitDataSetQueueOpName = "InitDataSetQueue";
constexpr auto kGetNext = "GetNext";
constexpr auto kDropoutGenMask = "DropoutGenMask";
constexpr auto kPrint = "Print";
const std::vector<std::string> AICPU_OPS = {kInitDataSetQueueOpName, kGetNext, kDropoutGenMask, kPrint};

std::shared_ptr<KernelBuildInfo> CreateKernelInfo(const std::vector<std::string> &inputs_format,
                                                  const std::vector<TypeId> &inputs_device_type,
                                                  const std::vector<std::string> &outputs_format,
                                                  const std::vector<TypeId> &outputs_device_type) {
  auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
  builder.SetInputsFormat(inputs_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsFormat(outputs_format);
  builder.SetOutputsDeviceType(outputs_device_type);
  builder.SetProcessor(AICPU);
  builder.SetKernelType(AICPU_KERNEL);
  builder.SetFusionType(OPAQUE);
  return builder.Build();
}

bool CheckIfExistAicpuMeta(const std::string &op_name) {
  if (std::find(AICPU_OPS.begin(), AICPU_OPS.end(), op_name) != AICPU_OPS.end()) {
    return false;
  }
  return true;
}

void AicpuMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  MS_LOG(INFO) << "AicpuMetadataInfo.";
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (CheckIfExistAicpuMeta(op_name)) {
    MS_LOG(DEBUG) << "Aicpu doesn't have metadata of op [" << op_name << "].";
    return;
  }

  if (op_name == kInitDataSetQueueOpName) {
    kernel_info_list->push_back(CreateKernelInfo({}, {}, {}, {}));
  }

  if (op_name == kGetNext) {
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(kernel_node); ++output_index) {
      outputs_format.emplace_back(kOpFormat_DEFAULT);
      outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
    }
    kernel_info_list->push_back(CreateKernelInfo({}, {}, outputs_format, outputs_type));
  }

  if (op_name == kDropoutGenMask) {
    kernel_info_list->push_back(CreateKernelInfo({kOpFormat_NCHW, kOpFormat_NCHW},
                                                 {kInt32->type_id(), kFloat16->type_id()}, {kOpFormat_NCHW},
                                                 {kUInt8->type_id()}));
  }

  if (op_name == kPrint) {
    std::vector<std::string> inputs_format;
    std::vector<TypeId> inputs_type;
    for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(kernel_node); ++input_index) {
      inputs_format.emplace_back(kOpFormat_DEFAULT);
      inputs_type.push_back(AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index));
    }
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(kernel_node); ++output_index) {
      outputs_format.emplace_back(kOpFormat_DEFAULT);
      outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
    }
    kernel_info_list->push_back(CreateKernelInfo(inputs_format, inputs_type, outputs_format, outputs_type));
  }

  if (kernel_info_list->empty()) {
    MS_LOG(INFO) << "Aicpu dose not has metadata of op[ " << op_name << "].";
  }
}
}  // namespace kernel
}  // namespace mindspore
