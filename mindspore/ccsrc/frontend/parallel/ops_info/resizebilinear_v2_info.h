/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESIZEBILINEAR_V2_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESIZEBILINEAR_V2_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ResizeBilinearV2Info : public OperatorInfo {
 public:
  ResizeBilinearV2Info(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ResizeBilinearCost>()) {}
  ~ResizeBilinearV2Info() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  void ReplaceNodeInputOrAttrs() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;
  Status CheckStrategyForDynamicShape(const StrategyPtr &strategy) override;

  std::vector<int64_t> size_;
  std::vector<int64_t> slice_size_;
  bool align_corners_ = false;
  bool need_exchange_overlap_ = false;

 private:
  Status InferRankBias();
  void InferOverlapSize();
  void InferScale();
  void InferNewOperatorAttrs();
  void InferCommunicationAttrs();
  void InferResizeBilinearV2Attrs();
  void InferReplaceGraph(const CNodePtr &cnode);
  int64_t InferOverlapLeftSizeByRankBias(int64_t rank_bias);
  int64_t InferOverlapRightSizeByRankBias(int64_t rank_bias);

  OperatorAttrs CreateNeighborExchangeV2Attrs();
  OperatorAttrs CreateParallelResizeBilinearAttrs();

  // rank_bias_ is the position of the current rank in the w dimension of the dev_matrix(have not split h dimension)
  int64_t rank_bias_ = 0;

  int64_t left_rank_bias_ = -1;
  int64_t right_rank_bias_ = -1;
  int64_t left_rank_id_ = -1;
  int64_t right_rank_id_ = -1;
  int64_t overlap_left_size_ = 0;
  int64_t overlap_right_size_ = 0;
  int64_t left_rank_overlap_right_size_ = 0;
  int64_t right_rank_overlap_left_size_ = 0;

  int64_t origin_in_w_shape_ = 1;
  int64_t origin_out_w_shape_ = 1;
  int64_t w_dimension_shard_num_ = 1;

  // the send_rank_ids_ or recv_rank_ids is an array with 8 rank ids, the order of index in the array is organized in
  // the following format(the 'R' is current rank)
  // +++++++++++++
  // | 7 | 0 | 1 |
  // +++++++++++++
  // | 6 | R | 2 |
  // +++++++++++++
  // | 5 | 4 | 3 |
  // +++++++++++++
  std::vector<int64_t> send_rank_ids_;  // 8 rank ids
  std::vector<int64_t> recv_rank_ids_;  // 8 rank ids

  // the send_lens_ or recv_lens_ is an array with 4 lens, the order in the array represents top, bottom, left, right
  std::vector<int64_t> send_lens_;  // [top, bottom, left, right]
  std::vector<int64_t> recv_lens_;  // [top, bottom, left, right]

  std::string all_to_all_group_;

  std::vector<int64_t> origin_image_size_;  // [H, W]
  int64_t src_start_w_ = 0;
  int64_t dst_start_w_ = 0;

  double w_scale_ = 1.0;  // the scale in w dimension, now only support to split w dimension
};

class ResizeNearestNeighborInfo : public ResizeBilinearV2Info {
 public:
  ResizeNearestNeighborInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                            const PrimitiveAttrs &attrs)
      : ResizeBilinearV2Info(name, inputs_shape, outputs_shape, attrs) {}
  ~ResizeNearestNeighborInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
};

}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESIZEBILINEAR_V2_INFO_H_
