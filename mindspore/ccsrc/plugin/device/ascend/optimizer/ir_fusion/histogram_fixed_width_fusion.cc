/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/histogram_fixed_width_fusion.h"

#include <memory>
#include <string>
#include <vector>

#include "mindspore/core/ops/structure_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace opt {
std::vector<std::string> HistogramFixedWidthFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimHistogramFixedWidth->name());
  return ret;
}

const BaseRef HistogramFixedWidthFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimHistogramFixedWidth, Xs});
}

const AnfNodePtr HistogramFixedWidthFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Begin to convert attr to input for node: " << node->DebugString();

  const auto &origin_prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(origin_prim);
  const auto &origin_attrs = origin_prim->attrs();

  constexpr auto kNbins = "nbins";
  if (origin_attrs.count(kNbins) == 0) {
    MS_LOG(DEBUG) << "Origin primitive: " << origin_prim->name() << "has no attr : " << kNbins;
    return node;
  }

  // Convert the specific attr to input and erase the specific attr.
  auto attr_value = origin_prim->GetAttr(kNbins);
  MS_EXCEPTION_IF_NULL(attr_value);
  if (attr_value->isa<Scalar>()) {
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto tensor_ptr = ScalarToTensor(attr_value->cast<ScalarPtr>());
    auto tensor_node = std::make_shared<ValueNode>(tensor_ptr);
    MS_EXCEPTION_IF_NULL(tensor_node);
    tensor_node->set_abstract(tensor_ptr->ToAbstract());
    tensor_node = kernel_graph->NewValueNode(tensor_node);
    kernel_graph->AddValueNodeToGraph(tensor_node);
    cnode->add_input(tensor_node);
    return cnode;
  }
  auto new_value_node = std::make_shared<ValueNode>(attr_value);
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(attr_value->ToAbstract());
  cnode->add_input(new_value_node);
  origin_prim->EraseAttr(kNbins);
  MS_LOG(DEBUG) << "End, new node: " << node->DebugString();
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
