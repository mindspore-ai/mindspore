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

#include <vector>
#include <memory>
#include "device/ascend/profiling/reporter/graph_desc_reporter.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace device {
namespace ascend {
void GraphDescReporter::ReportData() {
  for (const auto &node : cnode_list_) {
    if (AnfAlgo::GetKernelType(node) != TBE_KERNEL && AnfAlgo::GetKernelType(node) != AKG_KERNEL) {
      MS_LOG(WARNING) << "Skip non tbe kernel";
      continue;
    }
    std::vector<DataElement> input_data_list;
    std::vector<DataElement> output_data_list;
    MS_EXCEPTION_IF_NULL(node);
    auto op_name = node->fullname_with_scope();
    auto op_type = AnfAlgo::GetCNodeName(node);
    auto input_size = AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_size; ++i) {
      auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(node, i);
      auto input_node = input_node_with_index.first;
      auto input_index = input_node_with_index.second;
      DataElement element{};
      element.index_ = i;
      element.data_type_ = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
      element.data_format_ = AnfAlgo::GetOutputFormat(input_node, input_index);
      element.data_shape_ = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
      input_data_list.emplace_back(element);
    }

    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_size; ++i) {
      DataElement element{};
      element.index_ = i;
      element.data_type_ = AnfAlgo::GetOutputDeviceDataType(node, i);
      element.data_format_ = AnfAlgo::GetOutputFormat(node, i);
      element.data_shape_ = AnfAlgo::GetOutputDeviceShape(node, i);
      output_data_list.emplace_back(element);
    }

    auto graph_desc = std::make_shared<GraphDesc>(op_name, op_type, input_data_list, output_data_list);
    prof_desc_list_.emplace_back(graph_desc);
  }
  ReportAllLine();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
