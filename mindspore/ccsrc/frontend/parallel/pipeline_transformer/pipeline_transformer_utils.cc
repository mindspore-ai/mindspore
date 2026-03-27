/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pipeline_transformer/pipeline_transformer_utils.h"
#include <algorithm>
#include <vector>
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {

void SetMakeTupleAbstract(const CNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    return;
  }

  AbstractBasePtrList abstract_list;
  for (size_t i = 1; i < node->inputs().size(); i++) {
    abstract_list.emplace_back(node->input(i)->abstract());
  }
  auto abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  node->set_abstract(abs);
}

ValueListPtr ShapeDimensionsToValueList(const Shape &shape) {
  std::vector<ValuePtr> element;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(element),
                       [](int elem) { return MakeValue(elem); });
  return std::make_shared<ValueList>(element);
}

void SeparatePipelineParamBorder(const std::vector<AnfNodePtr> &nodes, bool send, std::vector<AnfNodePtr> *const params,
                                 std::vector<AnfNodePtr> *const borders) {
  std::vector<AnfNodePtr> real_comm_ops;
  if (send) {
    (void)std::transform(nodes.begin(), nodes.end(), std::back_inserter(real_comm_ops), [](const AnfNodePtr &n) {
      const auto &cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= INDEX_TWO) {
        return cnode;
      }
      const auto &real = cnode->input(INDEX_TWO)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(real);
      return real;
    });
  } else {
    real_comm_ops = nodes;
  }
  for (auto &node : real_comm_ops) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      (*params).push_back(node);
    } else {
      (*borders).push_back(node);
    }
  }
}

AbstractBasePtr GetRealAbstract(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &input = cnode->input(1);
    MS_EXCEPTION_IF_NULL(input);
    return input->abstract();
  }
  return node->abstract();
}

}  // namespace parallel
}  // namespace mindspore
