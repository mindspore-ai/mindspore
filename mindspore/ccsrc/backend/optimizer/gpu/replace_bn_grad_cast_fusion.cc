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
#include "backend/optimizer/gpu/replace_bn_grad_cast_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef ReplaceBNGradCastFusion::DefinePattern() const {
  VectorRef dy_cast = VectorRef({prim::kPrimCast, dy_});
  VectorRef fbn2g = VectorRef({prim::kPrimFusedBatchNormGrad, dy_cast, x_, scale_, mean_, var_});
  VectorRef tupleget = VectorRef({prim::kPrimTupleGetItem, fbn2g, index_});
  return tupleget;
}

const AnfNodePtr ReplaceBNGradCastFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  auto fbn2g = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);

  auto dy_after = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2g), 0);
  auto dy_before = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(dy_after), 0);
  auto x_ = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2g), 1);
  auto x_type = AnfAlgo::GetOutputInferDataType(x_, 0);
  // if x_type is fp32, the cast is nessery.
  if (x_type == kNumberTypeFloat32) {
    return nullptr;
  }
  auto scale = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2g), 2);
  auto mean = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2g), 3);
  auto var = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(fbn2g), 4);

  MS_EXCEPTION_IF_NULL(fbn2g);
  MS_EXCEPTION_IF_NULL(dy_after);
  MS_EXCEPTION_IF_NULL(dy_before);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(x_);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(var);
  std::vector<TypeId> outputs_type;
  std::vector<std::vector<size_t>> outputs_shape;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto outlist = GetRealNodeUsedList(graph, fbn2g);
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
  outputs_type.clear();
  outputs_shape.clear();
  manager->Replace(utils::cast<CNodePtr>(dy_after), utils::cast<CNodePtr>(dy_before));

  auto output_num = AnfAlgo::GetOutputTensorNum(fbn2g);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(fbn2g, i));
    outputs_shape.push_back(AnfAlgo::GetOutputInferShape(fbn2g, i));
  }
  outputs_type[0] = kNumberTypeFloat16;
  AnfAlgo::SetOutputInferTypeAndShape(outputs_type, outputs_shape, fbn2g.get());

  return node;
}
}  // namespace opt
}  // namespace mindspore
