/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/remove_fa_tensor_to_tuple_ops.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <algorithm>
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/array_ops.h"
#include "primitive/other_ops.h"
#include "primitive/arithmetic_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace {
bool IsFlashAttentionScoreNode(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimFlashAttentionScore)) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<std::string> RemoveFATensorToTupleOps::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimFlashAttentionScore->name()};
  return ret;
}

const BaseRef RemoveFATensorToTupleOps::DefinePattern() const {
  VarPtr resize = std::make_shared<CondVar>(IsFlashAttentionScoreNode);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({resize, inputs});
}

const AnfNodePtr RemoveFATensorToTupleOps::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto fa_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fa_cnode);

  auto fa_inputs = fa_cnode->inputs();
  for (size_t i = ops::kFlashAttentionScoreInputActualSeqQlenIndex;
       i <= ops::kFlashAttentionScoreInputActualSeqKVlenIndex; ++i) {
    auto input = fa_inputs.at(i + kSizeOne);
    if (IsValueNode<None>(input)) {
      continue;
    }

    if (IsPrimitiveCNode(input, prim::kPrimTensorToTuple)) {
      manager->SetEdge(fa_cnode, i + kSizeOne, input->cast<CNodePtr>()->input(kIndex1));
    }
  }
  return fa_cnode;
}
}  // namespace opt
}  // namespace mindspore
