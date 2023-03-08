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
#include "plugin/device/gpu/optimizer/reduce_precision_fusion.h"

#include <memory>
#include <string>
#include <vector>

#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
void ReducePrecision(const FuncGraphPtr &graph, const AnfNodePtr &node, size_t i, const TypeId &src_type,
                     const TypeId &cast_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i)};
  auto cast = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cast);
  prim->AddAttr(kAttrDstType, TypeIdToType(cast_type));
  auto cast_shape = {AnfAlgo::GetInputDeviceShape(node, i)};
  common::AnfAlgo::SetOutputInferTypeAndShape({cast_type}, cast_shape, cast.get());
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(node, i + 1, cast);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetInputsDeviceType({src_type});
  builder.SetOutputsDeviceType({cast_type});
  builder.SetKernelType(UNKNOWN_KERNEL_TYPE);
  builder.SetProcessor(kernel::Processor::CUDA);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
}
void ProcessTupleGetItem(const FuncGraphPtr &graph, const AnfNodePtr &node, size_t node_index, const TypeId &src_type,
                         const TypeId &cast_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, node_index);
  MS_EXCEPTION_IF_NULL(used_node_list);
  for (size_t i = 0; i < used_node_list->size(); i++) {
    auto used_node = used_node_list->at(i).first;
    auto used_node_index = used_node_list->at(i).second - 1;
    if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
      MS_LOG(EXCEPTION) << "TupleGetItem connect with TupleGetItem.";
    }
    ReducePrecision(graph, used_node, used_node_index, src_type, cast_type);
  }
}
}  // namespace
bool ReducePrecisionFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node != nullptr && node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      size_t output_num = AnfAlgo::GetOutputTensorNum(node);
      for (size_t i = 0; i < input_num; i++) {
        auto inferType = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i);
        auto deviceType = AnfAlgo::GetInputDeviceDataType(node, i);
        if (inferType == kNumberTypeInt64 && deviceType == kNumberTypeInt32) {
          ReducePrecision(graph, node, i, inferType, deviceType);
          MS_LOG(INFO) << "Reduce precision for [" << common::AnfAlgo::GetCNodeName(utils::cast<CNodePtr>(node))
                       << "] input " << i;
        }
      }
      for (size_t i = 0; i < output_num; i++) {
        auto inferType = common::AnfAlgo::GetOutputInferDataType(node, i);
        auto deviceType = AnfAlgo::GetOutputDeviceDataType(node, i);
        if (inferType != kNumberTypeInt64 || deviceType != kNumberTypeInt32) {
          continue;
        }
        auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, i);
        MS_EXCEPTION_IF_NULL(used_node_list);
        for (size_t j = 0; j < used_node_list->size(); j++) {
          auto used_node = used_node_list->at(j).first;
          auto used_node_index = used_node_list->at(j).second - 1;
          if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
            ProcessTupleGetItem(graph, used_node, used_node_index, deviceType, inferType);
          } else {
            ReducePrecision(graph, used_node, used_node_index, deviceType, inferType);
          }
        }
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
