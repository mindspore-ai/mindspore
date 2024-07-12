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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_SCORE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_SCORE_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <utility>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class FlashAttentionScoreInfo : public OperatorInfo {
 public:
  FlashAttentionScoreInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~FlashAttentionScoreInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReplaceNodeInputOrAttrs() override;
  void ReComputeBatchSplitFlagList() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

  int64_t input_layout() { return input_layout_; }
  int64_t s1_split_num() { return s1_split_num_; }
  bool kv_split() { return kv_split_; }
  int64_t head_num() { return head_num_; }
  bool real_shift_have_s1_dim() { return real_shift_have_s1_dim_; }
  bool real_shift_have_batch_dim() { return real_shift_have_batch_dim_; }
  bool is_attn_mask_compressed() { return is_attn_mask_compressed_; }
  bool attn_mask_have_n1_dim() { return attn_mask_have_n1_dim_; }
  bool attn_mask_have_batch_dim() { return attn_mask_have_batch_dim_; }
  std::vector<bool> is_input_passed() { return is_input_passed_; }
  size_t GetStrategyRealIndex(size_t index);
  Status InitAttrs();
  RankList GetSPRankList();
  int64_t GetActualSeqLengthSize();

 protected:
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status CheckStrategyForDynamicShape(const StrategyPtr &strategy) override;
  Status InferOutputTensorInfo() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputLayout();
  Status InferAsLossDivisorByLayout() override;
  Status InferMirrorOpsByLayout() override;
  Status InferSplitNumAndDevMatrixShapeByLayout();

 private:
  void UpdateDropoutGenMaskSliceShapeAndSeed(const CNodePtr &reshape_cnode);
  Status CheckInputInRingAttention();
  void InitIsInputPassed();
  Status InitQKVTensorMap();
  Status InitInputsTensorMap();
  Status InitSplittableInputs();
  Status InitAttnMaskSplittableInputs();
  Status InitExpectedStrategies();
  Status InitAttnMaskStrategies();
  Status InitQKVHeadAndSeqDimFromInputLayout();
  Status CheckStrategyExpected(const StrategyPtr &strategy);
  std::vector<int64_t> GetSplitIdAndRank();
  std::tuple<int64_t, int64_t> GetAttentionMaskAttrs(const int64_t split_id, const int64_t split_num);
  void LoadBalanceSplitAlongSeqDim(size_t input_index, GenerateGraph *gen_g, AnfNodePtr *split_node,
                                   AnfNodePtr *keep_node, AnfNodePtr *exchange_node);
  void LoadBalanceExchange(const int64_t all_gather_idx, const Group &group, const AnfNodePtr &input_node,
                           AnfNodePtr *exchange_node, GenerateGraph *gen_g);
  void GetFlashAttentionScoreOpNode(int64_t split_id, int64_t split_num, const AnfNodePtr &q,
                                    const AnfNodePtr &real_shift, const AnfNodePtr &drop_mask,
                                    const AnfNodePtr &attn_mask, AnfNodePtr *fa_op, GenerateGraph *gen_g);
  std::vector<std::pair<AnfNodePtr, int64_t>> ReplaceGraphGetInputNodes(const AnfNodePtr &q_split,
                                                                        const AnfNodePtr &real_shift_split,
                                                                        const AnfNodePtr &drop_mask_split,
                                                                        const AnfNodePtr &attn_mask_split,
                                                                        const AnfNodePtr &flash_attention_score_keep,
                                                                        const AnfNodePtr &flash_attention_score_target);
  Status ComputeReplaceGraphForLoadBalance(const CNodePtr &cnode);
  Status ReplaceActualSeqLenForSplitSeqInTnd(const CNodePtr &cnode);
  int64_t head_num_ = 1;
  float keep_prob_ = 1.0;
  float scale_value_ = 1.0;
  size_t qkv_batch_dim_;
  size_t qkv_head_dim_;
  size_t qkv_seq_dim_;
  int64_t pre_tokens_;
  int64_t next_tokens_;
  int64_t batch_split_num_;
  int64_t n1_split_num_;
  int64_t n2_split_num_;
  int64_t s1_split_num_;
  int64_t s2_split_num_;
  int64_t t1_split_num_;  // The split num of query's T-dim under 'TND'
  int64_t t2_split_num_;  // The split num of key and value's T=dim under 'TND'
  int64_t dev_matrix_batch_dim_;
  int64_t dev_matrix_n1_dim_;
  int64_t dev_matrix_s1_dim_;
  bool real_shift_have_s1_dim_ = false;     // true if real_shift and have s1 dim.
  bool real_shift_have_batch_dim_ = false;  // true if real_shift have batch dim
  bool attn_mask_have_batch_dim_ = false;   // true if attn_mask have batch dim.
  bool attn_mask_have_n1_dim_ = false;      // true if attn_mask have n1 dim.
  bool enable_load_balance_ = false;
  bool enable_ring_attention_ = false;
  bool enable_flash_sp_ = false;
  bool enable_ra_send_recv_ = false;
  int64_t input_layout_;  // "BSH": 0; "BNSD": 1;
  int64_t sparse_mode_;
  bool kv_split_ = false;
  bool is_attn_mask_compressed_ = false;
  bool need_update_op_attrs_mode_ = false;
  std::vector<bool> is_input_passed_;
  size_t real_input_size_ = 0;
  std::vector<Shape> splittable_inputs_;
  Strategies expect_strategies_;
  TensorLayout softmax_max_tensor_layout_;
  TensorLayout softmax_sum_tensor_layout_;
  TensorLayout softmax_out_tensor_layout_;
  TensorLayout attention_out_tensor_layout_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_SCORE_INFO_H_
