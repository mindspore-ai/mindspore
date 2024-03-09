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

 protected:
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;

 private:
  void UpdateDropoutGenMaskSliceShapeAndSeed(const CNodePtr &reshape_cnode);
  void InitIsInputPassed();
  void InitInputsTensorMap();
  void InitSplittableInputs();
  void InitExpectedStrategies();
  size_t GetStrategyRealIndex(size_t index);
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
  Status ComputeReplaceGraph(const CNodePtr &cnode);
  int64_t head_num_ = 1;
  float keep_prob_ = 1.0;
  float scale_value_ = 1.0;
  int64_t pre_tokens_;
  int64_t next_tokens_;
  std::string input_layout_;
  int64_t sparse_mode_;
  int64_t batch_split_num_;
  int64_t n1_split_num_;
  int64_t n2_split_num_;
  int64_t s1_split_num_;
  int64_t dev_matrix_batch_dim_;
  int64_t dev_matrix_n1_dim_;
  int64_t dev_matrix_s1_dim_;
  bool real_shift_have_s1_dim_ = false;     // true if real_shift and have s1 dim.
  bool real_shift_have_batch_dim_ = false;  // true if real_shift have batch dim
  bool attn_mask_have_batch_dim_ = false;   // true if attn_mask have batch dim.
  bool attn_mask_have_n1_dim_ = false;      // true if attn_mask have n1 dim.
  bool enable_load_balance_ = false;
  bool kv_split_ = false;
  bool is_attn_mask_compressed_ = false;
  std::vector<bool> is_input_passed_;
  std::vector<Shape> splittable_inputs_;
  Strategies expect_strategies_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_SCORE_INFO_H_
