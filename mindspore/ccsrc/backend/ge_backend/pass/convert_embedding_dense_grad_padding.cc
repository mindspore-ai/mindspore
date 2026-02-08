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

#include "backend/ge_backend/pass/convert_embedding_dense_grad_padding.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace opt {
std::vector<std::string> ConvertEmbeddingDenseGradPadding::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimEmbeddingDenseBackward->name()};
  return ret;
}

const BaseRef ConvertEmbeddingDenseGradPadding::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimEmbeddingDenseBackward, Xs});
}

const AnfNodePtr ConvertEmbeddingDenseGradPadding::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr convert_embedding_dense_grad_padding_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(convert_embedding_dense_grad_padding_cnode);

  AnfNodePtr num_weights_node = common::AnfAlgo::GetInputNode(convert_embedding_dense_grad_padding_cnode, kIndex2);
  AnfNodePtr padding_idx_node = common::AnfAlgo::GetInputNode(convert_embedding_dense_grad_padding_cnode, kIndex3);

  // get num_weights
  AbstractBasePtr num_weights_abstract = num_weights_node->abstract();
  ValuePtr num_weights_value = num_weights_abstract->GetValue();
  int64_t num_weights = GetValueWithCheck<int64_t>(num_weights_value);

  // get padding_idx
  AbstractBasePtr padding_idx_abstract = padding_idx_node->abstract();
  ValuePtr padding_idx_value = padding_idx_abstract->GetValue();
  int64_t padding_idx = GetValueWithCheck<int64_t>(padding_idx_value);

  // let padding_idx be positive
  int64_t positive_idx = padding_idx < 0 ? padding_idx + num_weights : padding_idx;

  auto value = std::make_shared<Int64Imm>(positive_idx);
  // create ValueNode
  auto value_node = NewValueNode(value);
  common::AnfAlgo::SetNodeInput(convert_embedding_dense_grad_padding_cnode, value_node, kIndex3);
  return node;
}
}  // namespace opt
}  // namespace mindspore
