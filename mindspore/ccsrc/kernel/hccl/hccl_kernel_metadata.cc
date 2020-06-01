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

#include "kernel/hccl/hccl_kernel_metadata.h"
#include <memory>
#include "utils/utils.h"
#include "kernel/hccl/hcom_util.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void HcclMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (op_name != kAllGather && op_name != kAllReduce && op_name != kBroadcast && op_name != kReduceScatter) {
    MS_LOG(DEBUG) << "Hccl does not have op [" << op_name << "]";
    return;
  }

  std::vector<std::string> inputs_format{};
  std::vector<TypeId> inputs_type{};
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(kernel_node); ++input_index) {
    inputs_format.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index));
    inputs_type.push_back(AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index));
  }

  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
  for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(kernel_node); ++output_index) {
    outputs_format.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(kernel_node, output_index));
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
  }
  auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
  builder.SetInputsFormat(inputs_format);
  builder.SetInputsDeviceType(inputs_type);
  builder.SetOutputsFormat(outputs_format);
  builder.SetOutputsDeviceType(outputs_type);
  builder.SetKernelType(HCCL_KERNEL);
  kernel_info_list->push_back(builder.Build());
}
}  // namespace kernel
}  // namespace mindspore
