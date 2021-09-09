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
#include "backend/optimizer/gpu/insert_cast_gpu.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
const std::unordered_set<std::string> kConv3DKernel = {prim::kPrimConv3DBackpropInput->name(),
                                                       prim::kPrimConv3DBackpropFilter->name(),
                                                       prim::kPrimConv3D->name(), prim::kPrimConv3DTranspose->name()};

void InsertCast(const FuncGraphPtr &graph, const AnfNodePtr &node, size_t i, const TypeId &src_type,
                const TypeId &cast_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i)};
  auto cast = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cast);
  auto cast_shape = {AnfAlgo::GetPrevNodeOutputInferShape(node, i)};
  AnfAlgo::SetOutputInferTypeAndShape({cast_type}, cast_shape, cast.get());
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(node, i + 1, cast);
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
    if (AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
      MS_LOG(EXCEPTION) << "TupleGetItem connect with TupleGetItem.";
    }
    InsertCast(graph, used_node, used_node_index, src_type, cast_type);
  }
}

bool InsertCastGPU::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  bool IsCasted = false;
  for (auto node : node_list) {
    if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
      continue;
    }

    if (kConv3DKernel.find(AnfAlgo::GetCNodeName(node)) == kConv3DKernel.end()) {
      continue;
    }

    size_t input_num = AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_num; i++) {
      auto inferType = AnfAlgo::GetPrevNodeOutputInferDataType(node, i);
      if (inferType == kNumberTypeFloat16) {
        InsertCast(graph, node, i, inferType, kNumberTypeFloat32);
        IsCasted = true;
        MS_LOG(INFO) << "Improve precision for [" << AnfAlgo::GetCNodeName(utils::cast<CNodePtr>(node)) << "] input "
                     << i;
      }
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; i++) {
      auto inferType = AnfAlgo::GetOutputInferDataType(node, i);
      if (inferType != kNumberTypeFloat16) {
        continue;
      }
      auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, i);
      MS_EXCEPTION_IF_NULL(used_node_list);
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        auto used_node_index = used_node_list->at(j).second - 1;
        if (AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
          ProcessTupleGetItem(graph, used_node, used_node_index, kNumberTypeFloat32, inferType);
        } else {
          InsertCast(graph, used_node, used_node_index, kNumberTypeFloat32, inferType);
        }
      }
    }

    if (IsCasted) {
      auto output_types = std::vector<TypeId>(output_num, kNumberTypeFloat32);
      std::vector<std::vector<size_t>> output_shapes;
      for (size_t output_index = 0; output_index < output_num; ++output_index) {
        std::vector<size_t> shape = AnfAlgo::GetOutputInferShape(node, output_index);
        (void)output_shapes.emplace_back(shape);
      }
      AnfAlgo::SetOutputInferTypeAndShape(output_types, output_shapes, node.get());
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
