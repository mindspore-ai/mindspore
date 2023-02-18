/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV2D_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV2D_INFO_H_

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
class Conv2DInfo : public OperatorInfo {
 public:
  Conv2DInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<BatchParallelCost>()) {}
  ~Conv2DInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  void ReComputeBatchSplitFlagList() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrsBase();
  virtual Status CheckAttrsBase();
  virtual std::vector<int64_t> GetStrideAttr();
  virtual std::vector<int64_t> GetDilationAttr();
  Status GetAttrs() override;
  Status CheckStrategyBase(const StrategyPtr &strategy);
  Status CheckHWStrategyBase(int64_t h_strategy, int64_t w_strategy) const;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  void InferAdjacentRankInfo();
  std::vector<int64_t> GetAdjacentRankIdsAndBiases(int64_t rank_id, int64_t dimension);
  void InferOverlapSize();
  void CheckHDimensionOverlapSizeNonNegative();
  void CheckWDimensionOverlapSizeNonNegative();
  void CheckOverlapSizeNonNegative();
  void InferOverlapSizeForHDim();
  void InferOverlapSizeForWDim();
  void InferNewOperatorAttrs();
  void InferSendRankIds();
  void InferRecvRankIds();
  void InferCommunicationAttrs();
  virtual std::string ReplaceNodeName() const;
  AnfNodePtr GenerateConv2DNode(const AnfNodePtr &new_input, const CNodePtr &cnode);
  OperatorAttrs CreateNeighborExchangeV2Attrs();
  OperatorAttrs CreateConv2DAttrs();
  virtual void ComputeReplaceGraph(const CNodePtr &cnode);

  int64_t out_channel_ = 1;
  std::vector<int64_t> kernel_size_;               // two integers
  std::vector<int64_t> kernel_size_use_dilation_;  // two integers, it is dilation * (kernel_size - 1) + 1
  int64_t mode_ = 1;
  int64_t pad_mode_ = 0;           // "pad": 0; "same": 1; "valid": 2;
  std::vector<int64_t> pad_list_;  // four integers
  std::vector<int64_t> stride_;    // four integers
  std::vector<int64_t> dilation_;  // four integers
  int64_t group_ = 1;
  std::string format_;
  bool out_channel_shard_ = false;
  int64_t new_out_channel_ = 1;
  std::vector<int64_t> new_pad_list_;

  bool w_dim_need_exchange_overlap_ = false;
  bool h_dim_need_exchange_overlap_ = false;
  int64_t h_rank_bias_ = 0;        // the bias of current rank in h dimension of device matrix
  int64_t w_rank_bias_ = 0;        // the bias of current rank in w dimension of device matrix
  int64_t top_rank_bias_ = -1;     // the bias of top rank in h dimension of device matrix
  int64_t bottom_rank_bias_ = -1;  // the bias of bottom rank in h dimension of device matrix
  int64_t left_rank_bias_ = -1;    // the bias of left rank in w dimension of device matrix
  int64_t right_rank_bias_ = -1;   // the bias of right rank in w dimension of device matrix

  // 8 adjacent ranks
  int64_t top_rank_id_ = -1;
  int64_t top_right_rank_id_ = -1;
  int64_t right_rank_id_ = -1;
  int64_t bottom_right_rank_id_ = -1;
  int64_t bottom_rank_id_ = -1;
  int64_t bottom_left_rank_id_ = -1;
  int64_t left_rank_id_ = -1;
  int64_t top_left_rank_id_ = -1;

  // overlap sizes for h dimension
  int64_t overlap_top_size_ = 0;
  int64_t overlap_bottom_size_ = 0;
  int64_t top_rank_overlap_bottom_size_ = 0;
  int64_t bottom_rank_overlap_top_size_ = 0;

  // overlap sizes for w dimension
  int64_t overlap_left_size_ = 0;
  int64_t overlap_right_size_ = 0;
  int64_t left_rank_overlap_right_size_ = 0;
  int64_t right_rank_overlap_left_size_ = 0;

  int64_t h_dimension_shard_num_ = 1;
  int64_t w_dimension_shard_num_ = 1;
  Shape input_slice_shape_;

  // the send_rank_ids_ or recv_rank_ids is an array with 8 rank ids, the order of index in the array is organized in
  // the following format(the 'R' is current rank), the invalid rank fill -1
  // +++++++++++++
  // | 7 | 0 | 1 |
  // +++++++++++++
  // | 6 | R | 2 |
  // +++++++++++++
  // | 5 | 4 | 3 |
  // +++++++++++++
  std::vector<int64_t> send_rank_ids_;
  std::vector<int64_t> recv_rank_ids_;

  // the send_lens_ or recv_lens_ is an array with 4 lens, the order in the array represents top, bottom, left, right
  std::vector<int64_t> send_lens_;
  std::vector<int64_t> recv_lens_;
  std::string all_to_all_group_;

  GenerateGraph gen_g_ = GenerateGraph(attrs_);

  virtual Status CheckHWStrategy(int64_t h_strategy, int64_t w_strategy);
  virtual void InferNewPadList();
  virtual int64_t ComputeOverlapTopSizeByRankBias(int64_t rank_bias);
  virtual int64_t ComputeOverlapBottomSizeByRankBias(int64_t rank_bias);
  virtual int64_t ComputeOverlapLeftSizeByRankBias(int64_t rank_bias);
  virtual int64_t ComputeOverlapRightSizeByRankBias(int64_t rank_bias);

 private:
  Status CheckHWStrategyValidMode(int64_t h_strategy, int64_t w_strategy);
  Status CheckHWStrategyPadModeByDimension(int64_t strategy, int64_t dimension_id);
  Status CheckHWStrategyPadMode(int64_t h_strategy, int64_t w_strategy);
  void AdjustPadList();
  bool pad_list_adjusted_ = false;
};

class Conv2DBackpropInputInfo : public Conv2DInfo {
 public:
  Conv2DBackpropInputInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : Conv2DInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~Conv2DBackpropInputInfo() override = default;
  void UpdateOutShape();
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status GetAttrs() override;
  Status GetOutShape();
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;  // can not use OperatorInfo::InferMirrorOps(), since the 'out_shape' is not tensor

  Status CheckHWStrategy(int64_t h_strategy, int64_t w_strategy) override;
  void InferNewPadList() override;
  void InferNewPadListByDimension(const std::string &dimension);
  int64_t ComputeOverlapTopSizeByRankBias(int64_t rank_bias) override;
  int64_t ComputeOverlapBottomSizeByRankBias(int64_t rank_bias) override;
  int64_t ComputeOverlapLeftSizeByRankBias(int64_t rank_bias) override;
  int64_t ComputeOverlapRightSizeByRankBias(int64_t rank_bias) override;

 private:
  Shape out_shape_;
  Shape out_slice_shape_;
};

class Conv2DTransposeInfo : public Conv2DBackpropInputInfo {
 public:
  Conv2DTransposeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : Conv2DBackpropInputInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~Conv2DTransposeInfo() override = default;
};

constexpr size_t IN_CHANNEL_INDEX = 1;
using Conv2DBackpropInputInfoPtr = std::shared_ptr<Conv2DBackpropInputInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV2D_INFO_H_
