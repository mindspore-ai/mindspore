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

#include "frontend/parallel/graph_util/solve_flash_attention_score_for_seqpipe.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <set>

#include "include/utils/anfalgo.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/core/include/ir/value.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "primitive/arithmetic_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace parallel {
namespace {
typedef struct FASparseInfo {
  FASparseInfo(int64_t pre_tokens_, int64_t next_tokens_, int64_t sparse_mode_, int64_t q_seq_len_, int64_t kv_seq_len_)
      : pre_tokens(pre_tokens_),
        next_tokens(next_tokens_),
        sparse_mode(sparse_mode_),
        q_seq_len(q_seq_len_),
        kv_seq_len(kv_seq_len_) {}
  bool operator==(const FASparseInfo &cmp) {
    return pre_tokens == cmp.pre_tokens && next_tokens == cmp.next_tokens && sparse_mode == cmp.sparse_mode &&
           q_seq_len == cmp.q_seq_len && kv_seq_len == cmp.kv_seq_len;
  }
  bool operator!=(const FASparseInfo &cmp) { return !(*this == cmp); }

  int64_t pre_tokens;
  int64_t next_tokens;
  int64_t sparse_mode;
  int64_t q_seq_len;
  int64_t kv_seq_len;
} FASparseInfo;

int64_t InferSeqDimByInputLayout(int64_t input_layout) {
  if (ops::layoutMap.find(input_layout) == ops::layoutMap.end()) {
    MS_LOG(EXCEPTION) << "Invalid input layout: " << input_layout;
  }
  auto input_layout_str = ops::layoutMap.at(input_layout);
  auto dim = input_layout_str.find('S');
  if (dim != std::string::npos) {
    return dim;
  }
  dim = input_layout_str.find('T');
  if (dim != std::string::npos) {
    return dim;
  }
  MS_LOG(EXCEPTION) << "Cannot find seq dim in layout: " << input_layout_str;
}

bool IsFlashAttentionVarLen(const CNodePtr &fa_cnode) {
  MS_EXCEPTION_IF_NULL(fa_cnode);
  auto input_layout = GetInputValueFromCNode<int64_t>(fa_cnode, ops::kFlashAttentionScoreInputLayoutIndex + 1);
  if (ops::layoutMap.find(input_layout) == ops::layoutMap.end()) {
    MS_LOG(DEBUG) << "The input_layout " << input_layout << " is not support yet.";
    return false;
  }
  auto input_layout_str = ops::layoutMap.at(input_layout);
  return input_layout_str.find('T') != std::string::npos;
}

FASparseInfo ExtractSparseInfoFromFlashAttentionScore(const CNodePtr &fa_cnode) {
  MS_EXCEPTION_IF_NULL(fa_cnode);
  if (!IsPrimitiveCNode(fa_cnode)) {
    MS_LOG(EXCEPTION) << "Failed to extract sparse info from node: " << fa_cnode->DebugString();
  }
  auto pre_tokens = GetInputValueFromCNode<int64_t>(fa_cnode, ops::kFlashAttentionScoreInputPreTokensIndex + 1);
  auto next_tokens = GetInputValueFromCNode<int64_t>(fa_cnode, ops::kFlashAttentionScoreInputNextTokensIndex + 1);
  auto sparse_mode = GetInputValueFromCNode<int64_t>(fa_cnode, ops::kFlashAttentionScoreInputSparseModeIndex + 1);
  auto input_layout = GetInputValueFromCNode<int64_t>(fa_cnode, ops::kFlashAttentionScoreInputLayoutIndex + 1);

  auto query_shape =
    common::AnfAlgo::GetOutputInferShape(fa_cnode->input(ops::kFlashAttentionScoreInputQueryIndex + 1), 0);
  auto key_shape = common::AnfAlgo::GetOutputInferShape(fa_cnode->input(ops::kFlashAttentionScoreInputKeyIndex + 1), 0);
  auto seq_dim = InferSeqDimByInputLayout(input_layout);
  auto q_seq_len = query_shape[seq_dim];
  auto kv_seq_len = key_shape[seq_dim];
  return FASparseInfo(pre_tokens, next_tokens, sparse_mode, q_seq_len, kv_seq_len);
}

CNodePtr CreateTensorToScalarCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input) {
  auto tensor_to_scalar_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimTensorToScalar), input});
  tensor_to_scalar_cnode->set_abstract(std::make_shared<abstract::AbstractScalar>(TypeIdToType(kNumberTypeInt64)));
  return tensor_to_scalar_cnode;
}

std::optional<FASparseInfo> SolveFuncGraphIfContainFlashAttentionScore(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &all_nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  CNodePtrList fa_cnode_list;
  std::for_each(all_nodes.begin(), all_nodes.end(), [&fa_cnode_list](const AnfNodePtr &node) {
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      fa_cnode_list.push_back(node->cast<CNodePtr>());
    }
  });
  if (fa_cnode_list.empty()) {
    return std::nullopt;
  }
  auto sparse_fa_info = ExtractSparseInfoFromFlashAttentionScore(fa_cnode_list.front());
  for (auto iter = fa_cnode_list.begin() + 1; iter != fa_cnode_list.end(); ++iter) {
    auto cur_sparse_fa_info = ExtractSparseInfoFromFlashAttentionScore((*iter)->cast<CNodePtr>());
    if (cur_sparse_fa_info != sparse_fa_info) {
      return std::nullopt;
    }
  }
  if (IsFlashAttentionVarLen(fa_cnode_list.front())) {
    MS_LOG(DEBUG) << "There is no need to replace sparse parameter for FlashAttentionScoreVarLen, skip this pass.";
    return std::nullopt;
  }

  auto new_pre_tokens_tensor = std::make_shared<Parameter>(func_graph);
  auto new_next_tokens_tensor = std::make_shared<Parameter>(func_graph);
  auto new_sparse_mode_tensor = std::make_shared<Parameter>(func_graph);
  func_graph->add_parameter(new_pre_tokens_tensor);
  func_graph->add_parameter(new_next_tokens_tensor);
  func_graph->add_parameter(new_sparse_mode_tensor);
  auto new_pre_tokens = CreateTensorToScalarCNode(func_graph, new_pre_tokens_tensor);
  auto new_next_tokens = CreateTensorToScalarCNode(func_graph, new_next_tokens_tensor);
  auto new_sparse_mode = CreateTensorToScalarCNode(func_graph, new_sparse_mode_tensor);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &fa_cnode : fa_cnode_list) {
    manager->SetEdge(fa_cnode, ops::kFlashAttentionScoreInputPreTokensIndex + 1, new_pre_tokens);
    manager->SetEdge(fa_cnode, ops::kFlashAttentionScoreInputNextTokensIndex + 1, new_next_tokens);
    manager->SetEdge(fa_cnode, ops::kFlashAttentionScoreInputSparseModeIndex + 1, new_sparse_mode);
  }
  return sparse_fa_info;
}
}  // namespace

void SolveFASparseForSeqPipe(const CNodePtrList &call_cnode_list, const size_t seq_chunk_num) {
  if (call_cnode_list.empty()) {
    MS_LOG(DEBUG) << "call_cnode_list is empty, skip SolveFASparseForSeqPipe";
    return;
  }

  // (sub_graph, (seq_chunk, call_cnode_list))
  std::unordered_map<FuncGraphPtr, std::unordered_map<int64_t, std::set<CNodePtr>>> func_graph_map;
  for (const auto &call_cnode : call_cnode_list) {
    if (!IsValueNode<FuncGraph>(call_cnode->input(0)) || !call_cnode->HasPrimalAttr(SEQ_CHUNK)) {
      continue;
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(call_cnode->input(kIndex0));
    MS_EXCEPTION_IF_NULL(func_graph);
    auto seq_chunk = GetValue<int64_t>(call_cnode->GetPrimalAttr(SEQ_CHUNK));
    func_graph_map[func_graph][seq_chunk].insert(call_cnode);
  }
  for (const auto &sub_graph_pair : func_graph_map) {
    auto sub_graph = sub_graph_pair.first;
    auto sparse_info = SolveFuncGraphIfContainFlashAttentionScore(sub_graph);
    if (!sparse_info.has_value()) {
      continue;
    }
    auto seq_chunk_call_nodes_map = sub_graph_pair.second;
    for (const auto &seq_chunk_call_nodes_pair : seq_chunk_call_nodes_map) {
      auto seq_chunk = seq_chunk_call_nodes_pair.first;
      int64_t new_pre_tokens;
      int64_t new_next_tokens;
      int64_t new_sparse_mode;
      ComputeSparseInfoForFlashAttentionScore(
        sparse_info->sparse_mode, sparse_info->pre_tokens, sparse_info->next_tokens, seq_chunk, seq_chunk_num,
        sparse_info->q_seq_len, sparse_info->kv_seq_len, &new_sparse_mode, &new_pre_tokens, &new_next_tokens);
      auto call_cnodes = seq_chunk_call_nodes_pair.second;
      auto new_pre_tokens_node = CreateInt32Tensor(new_pre_tokens, true);
      auto new_next_tokens_node = CreateInt32Tensor(new_next_tokens, true);
      auto new_sparse_mode_node = CreateInt32Tensor(new_sparse_mode, true);
      for (const auto &call_cnode : call_cnodes) {
        call_cnode->add_input(new_pre_tokens_node);
        call_cnode->add_input(new_next_tokens_node);
        call_cnode->add_input(new_sparse_mode_node);
      }
      MS_LOG(DEBUG) << "For seq_chunk " << seq_chunk << ", replace pre_tokens from " << sparse_info->pre_tokens
                    << " to " << new_pre_tokens << ", next_tokens from " << sparse_info->next_tokens << " to "
                    << new_next_tokens << ", "
                    << ", sparse_mode from " << sparse_info->sparse_mode << " to " << new_sparse_mode;
    }
  }
  return;
}
}  // namespace parallel
}  // namespace mindspore
