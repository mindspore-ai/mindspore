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

#include "plugin/device/gpu/optimizer/remove_format_transform_pair.h"
#include <memory>
#include "mindspore/core/ops/array_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveFormatTransformPair::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  VarPtr Z = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(Y);
  MS_EXCEPTION_IF_NULL(Z);
  VectorRef transpose1 = VectorRef({prim::kPrimTranspose, X, Y});
  VectorRef transpose2 = VectorRef({prim::kPrimTranspose, transpose1, Z});
  return transpose2;
}

const AnfNodePtr RemoveFormatTransformPair::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Process node:" << node->fullname_with_scope();
  auto transpose2 = utils::cast<CNodePtr>(node);
  auto transpose1 = utils::cast<CNodePtr>(common::AnfAlgo::GetInputNode(transpose2, 0));
  auto prim_name = prim::kPrimTranspose->name();
  if (common::AnfAlgo::GetCNodeName(transpose1) != prim_name ||
      common::AnfAlgo::GetCNodeName(transpose2) != prim_name) {
    MS_LOG(EXCEPTION) << "The  pattern is not transpose pair, "
                      << "node:" << common::AnfAlgo::GetCNodeName(transpose2)
                      << " node input:" << common::AnfAlgo::GetCNodeName(transpose1);
  }
  // If transpose operator used by more than one other operators, it cant not be deleted directly.
  if (IsUsedByOthers(graph, transpose1)) {
    MS_LOG(DEBUG) << "The transpose node [" << transpose1->fullname_with_scope()
                  << "] is used by more than one other operators.";
    return nullptr;
  }

  auto transpose1_input_format = AnfAlgo::GetInputFormat(transpose1, 0);
  auto transpose2_output_format = AnfAlgo::GetOutputFormat(transpose2, 0);
  if (transpose1_input_format != transpose2_output_format) {
    MS_LOG(DEBUG) << "The input format of the first transpose is different from the output format of the second "
                     "transpose, can't remove this transpose pair.";
    return nullptr;
  }
  auto perm1 = common::AnfAlgo::GetInputNode(transpose1, 1);
  auto perm2 = common::AnfAlgo::GetInputNode(transpose2, 1);
  auto perm1_value = perm1->cast<ValueNodePtr>()->value();
  auto perm2_value = perm2->cast<ValueNodePtr>()->value();
  auto perm1_vec = CheckAndConvertUtils::CheckTensorIntValue("permutation1", perm1_value, prim_name);
  auto perm2_vec = CheckAndConvertUtils::CheckTensorIntValue("permutation2", perm2_value, prim_name);
  auto dim = perm1_vec.size();
  for (size_t i = 0; i < dim; i++) {
    MS_EXCEPTION_IF_CHECK_FAIL(i < perm2_vec.size(), "perm is out of bound");
    auto index = perm2_vec[i] < 0 ? perm2_vec[i] + dim : perm2_vec[i];
    MS_EXCEPTION_IF_CHECK_FAIL(index < dim, "perm is out of bound");
    auto axis = perm1_vec[index] < 0 ? perm1_vec[index] + dim : perm1_vec[index];
    if (static_cast<size_t>(axis) != i) {
      MS_LOG(DEBUG) << "Permutation changes, can't remove this transpose pair";
      return nullptr;
    }
  }
  auto transpose1_input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(transpose1), 0);
  MS_EXCEPTION_IF_NULL(transpose1_input_node);
  return transpose1_input_node;
}
}  // namespace opt
}  // namespace mindspore
