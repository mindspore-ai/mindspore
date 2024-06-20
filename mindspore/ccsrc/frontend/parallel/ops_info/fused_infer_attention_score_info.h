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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FUSED_INFER_ATTENTION_SCORE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FUSED_INFER_ATTENTION_SCORE_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <tuple>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class FusedInferAttentionScoreInfo : public OperatorInfo {
 public:
  // Generate all strategies and the corresponding cost for this MatMul operator
  FusedInferAttentionScoreInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                               const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~FusedInferAttentionScoreInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReComputeBatchSplitFlagList() override;
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 private:
  bool CheckStrategy(int64_t strategy, int64_t true_value, const std::string &dim_name, const std::string &input_name);
  void GenerateExpectStrategies();
  bool CheckStrategyOnIndex(int64_t strategy, int64_t true_value, const std::string &dim_name,
                            const std::string &input_name);
  std::tuple<int64_t, int64_t> GetAttentionMaskAttrs(const int64_t split_id, const int64_t split_num);
  int64_t GetSplitIdAndRank();
  void SetOptionalInputs();
  void InferOptionalTensorMap();
  Status CheckQueryStrategy(const NewStrategies &stra);
  void SplitKVSequenceGraph(const Group &group, GenerateGraph *gen_g, AnfNodePtr *fused_attention_score,
                            AnfNodePtr *output);
  Status ComputeReplaceGraphForSplitKVSeq(const CNodePtr &cnode);
  int64_t head_num_;
  int64_t kv_head_num_;
  int64_t input_layout_;
  int64_t dp_;
  int64_t mp_;
  int64_t sp_;
  int64_t sparse_mode_;
  int64_t pre_tokens_;
  int64_t next_tokens_;
  bool softmax_lse_flag_ = false;
  int64_t dev_matrix_batch_dim_;
  int64_t dev_matrix_s1_dim_;
  int64_t dev_matrix_n1_dim_;
  size_t expect_strategies_size_;
  std::vector<Shape> optional_tensor_map_ = {{}, {}, {}, {}, {}, {2}, {2}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  std::vector<Shape> optional_op_strategies_ = {{},  {},  {},  {1, 0, 0, 0}, {1, 0, 0}, {1}, {1}, {0}, {0},
                                                {0}, {0}, {0}, {0},          {0},       {0}, {0}, {0}};
  std::vector<bool> optional_inputs_;
  size_t atten_mask_rank_ = 0;
  size_t pse_shift_rank_ = 0;
  std::vector<Shape> expect_strategies_;
  bool is_ifa_ = false;
  bool enable_ring_attention_ = false;
  bool is_attn_mask_compressed_ = false;
  bool need_update_op_attrs_mode_ = false;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FUSED_INFER_ATTENTION_SCORE_INFO_H_
