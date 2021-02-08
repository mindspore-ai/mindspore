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

#include "backend/kernel_compiler/aicpu/aicpu_kernel_metadata.h"
#include <memory>
#include <string>
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void AicpuMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  MS_LOG(INFO) << "AicpuMetadataInfo.";
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kAICPU);
  if (op_info_ptr == nullptr) {
    MS_LOG(DEBUG) << "Aicpu does not have op [" << op_name << "]";
    return;
  }
  // For compatibility with the current framework
  if (op_name == kPrint || op_name == kGetNext || op_name == kPack || op_name == kMeshgrid) {
    std::vector<std::string> inputs_format{};
    std::vector<TypeId> inputs_type{};
    if (op_name == kPrint || op_name == kPack || op_name == kMeshgrid) {
      size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
      for (size_t input_index = 0; input_index < input_num; ++input_index) {
        inputs_format.emplace_back(kOpFormat_DEFAULT);
        inputs_type.push_back(AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index));
      }
    }
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      outputs_format.emplace_back(kOpFormat_DEFAULT);
      outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
    }
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    builder.SetInputsFormat(inputs_format);
    builder.SetInputsDeviceType(inputs_type);
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_type);
    builder.SetProcessor(AICPU);
    builder.SetKernelType(AICPU_KERNEL);
    builder.SetFusionType(OPAQUE);
    kernel_info_list->push_back(builder.Build());
    return;
  }
  if (!ParseMetadata(kernel_node, op_info_ptr, AICPU, kernel_info_list)) {
    MS_LOG(WARNING) << "Aicpu parsed metadata op [" << op_name << "] failed";
    return;
  }
}
}  // namespace kernel
}  // namespace mindspore
