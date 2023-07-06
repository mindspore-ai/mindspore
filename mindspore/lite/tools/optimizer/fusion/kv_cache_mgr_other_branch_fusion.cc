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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/kv_cache_mgr_other_branch_fusion.h"
#include <memory>
#include <vector>
#include "schema/inner/model_generated.h"
#include "ops/affine.h"
#include "src/common/log_adapter.h"
#include "ops/splice.h"
#include "ops/mat_mul.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "ops/math_ops.h"
#include "ops/comparison_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/fusion/kv_cache_mgr.h"
#include "ops/add.h"
#include "ops/expand_dims.h"
#include "ops/mul.h"
#include "ops/make_tuple.h"
#include "ops/concat.h"
#include "ops/assign.h"

namespace mindspore::opt {
const BaseRef KVCacheMgrOtherBranchFusion::DefinePattern() const {
  if (!InitVar()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref({is_mul, input_1_key_, input_0_concat_});

  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, input_2_key_past_, mul_ref});

  return add_ref;
}

bool KVCacheMgrOtherBranchFusion::InputIsConcat(const EquivPtr &equiv) const {
  auto input_0_concat_cnode = utils::cast<CNodePtr>((*equiv)[input_0_concat_]);
  MS_ASSERT(input_0_concat_cnode != nullptr);
  auto concat_prim = ops::GetOperator<ops::Concat>(input_0_concat_cnode->input(0));
  if (concat_prim == nullptr) {
    MS_LOG(INFO) << "Concat prim is nullptr or cnode is not Concat CNode.";
    return false;
  }
  return true;
}

CNodePtr KVCacheMgrOtherBranchFusion::CreateKVCacheMgrNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto kv_cache_prim = std::make_shared<ops::KVCacheMgr>();
  MS_CHECK_TRUE_RET(kv_cache_prim != nullptr, nullptr);
  auto kv_cache_prim_c = kv_cache_prim->GetPrim();
  MS_CHECK_TRUE_RET(kv_cache_prim_c != nullptr, nullptr);

  auto input_2_past_node = utils::cast<AnfNodePtr>((*equiv)[input_2_key_past_]);
  MS_ASSERT(input_2_past_node != nullptr);
  auto input_1_cur_node = utils::cast<AnfNodePtr>((*equiv)[input_1_key_]);
  MS_ASSERT(input_1_cur_node != nullptr);
  auto input_0_concat_node = utils::cast<AnfNodePtr>((*equiv)[input_0_concat_]);
  MS_ASSERT(input_0_concat_node != nullptr);

  auto kv_cache_cnode =
    func_graph->NewCNode(kv_cache_prim_c, {input_2_past_node, input_1_cur_node, input_0_concat_node});
  kv_cache_cnode->set_abstract(input_2_past_node->abstract()->Clone());

  return kv_cache_cnode;
}

bool KVCacheMgrOtherBranchFusion::InitVar() const {
  input_0_concat_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_0_concat_ != nullptr, false);
  input_1_key_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_1_key_ != nullptr, false);
  input_2_key_past_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_2_key_past_ != nullptr, false);
  return true;
}

const AnfNodePtr KVCacheMgrOtherBranchFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "KVCacheMgrOtherBranchFusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }

  if (!InputIsConcat(equiv)) {
    MS_LOG(INFO) << "Not is KVCache Pattern.";
    return nullptr;
  }

  auto cnode = CreateKVCacheMgrNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "new kvcache node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace mindspore::opt
