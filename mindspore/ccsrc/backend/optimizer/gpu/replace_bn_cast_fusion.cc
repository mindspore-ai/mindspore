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
#include "backend/optimizer/gpu/replace_bn_cast_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef ReplaceBNCastFusion::DefinePattern() const {
  VectorRef in_cast = VectorRef({prim::kPrimCast, x_});
  VectorRef fbn2 = VectorRef({prim::kPrimFusedBatchNorm, in_cast, scale_, bias_, mean_, var_});
  VectorRef tupleget = VectorRef({prim::kPrimTupleGetItem, fbn2, index_});
  return tupleget;
}

const AnfNodePtr ReplaceBNCastFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  auto fbn2 = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto x_after = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2), 0);
  auto x_before = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(x_after), 0);
  auto scale = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2), 1);
  auto bias = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2), 2);
  auto mean = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2), 3);
  auto var = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2), 4);

  MS_EXCEPTION_IF_NULL(fbn2);
  MS_EXCEPTION_IF_NULL(x_after);
  MS_EXCEPTION_IF_NULL(x_before);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(bias);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(var);
  std::vector<TypeId> outputs_type;
  std::vector<std::vector<size_t>> outputs_shape;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto outlist = GetRealNodeUsedList(graph, fbn2);
  for (size_t i = 0; i < outlist->size(); i++) {
    auto index_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(outlist->at(i).first), 1);
    auto value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    int item_idx = GetValue<int>(value_node->value());
    if (item_idx == 0) {
      auto cast = GetRealNodeUsedList(graph, outlist->at(i).first);
      if (AnfAlgo::GetCNodeName(cast->at(0).first) != "Cast") {
        return nullptr;
      }
      manager->Replace(utils::cast<CNodePtr>(cast->at(0).first), utils::cast<CNodePtr>(outlist->at(i).first));
      outputs_type.push_back(kNumberTypeFloat16);
      outputs_shape.push_back(AnfAlgo::GetOutputInferShape(outlist->at(i).first, 0));
      AnfAlgo::SetOutputInferTypeAndShape(outputs_type, outputs_shape, outlist->at(i).first.get());
    }
  }

  manager->Replace(utils::cast<CNodePtr>(x_after), utils::cast<CNodePtr>(x_before));
  outputs_type.clear();
  outputs_shape.clear();
  auto output_num = AnfAlgo::GetOutputTensorNum(fbn2);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(fbn2, i));
    outputs_shape.push_back(AnfAlgo::GetOutputInferShape(fbn2, i));
  }
  outputs_type[0] = kNumberTypeFloat16;
  AnfAlgo::SetOutputInferTypeAndShape(outputs_type, outputs_shape, fbn2.get());
  return node;
}
}  // namespace opt
}  // namespace mindspore
