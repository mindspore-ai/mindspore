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
#include <unordered_map>
#include <vector>

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

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t) override;
  Status SetCostUnderStrategy(const StrategyPtr &) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrsBase();
  Status GetAttrs() override;
  Status CheckStrategyBase(const StrategyPtr &strategy);
  Status CheckHWStrategyBase(int64_t h_strategy, int64_t w_strategy) const;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferRankBias();
  void InferOverlapSize();
  void InferNewOperatorAttrs();
  void InferSendRecvFlag();
  void InferOverlapShapes();
  void InferStridedSliceAttrs();
  std::string ReplaceNodeName() const;
  AnfNodePtr GenerateConv2DNode(const AnfNodePtr &new_input, const CNodePtr &cnode);
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  OperatorAttrs CreateNeighborExchangeAttrs(const CNodePtr &cnode);
  OperatorAttrs CreateConv2DAttrs();
  void ComputeReplaceGraph(const CNodePtr &cnode);

  int64_t out_channel_ = 1;
  std::vector<int64_t> kernel_size_;  // two integers
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

  bool need_exchange_overlap_ = false;
  int64_t rank_bias_ = 0;
  int64_t left_rank_bias_ = -1;
  int64_t right_rank_bias_ = -1;
  int64_t left_rank_id_ = -1;
  int64_t right_rank_id_ = -1;
  int64_t overlap_left_size_ = 0;
  int64_t overlap_right_size_ = 0;
  int64_t left_rank_overlap_left_size_ = 0;
  int64_t left_rank_overlap_right_size_ = 0;
  int64_t right_rank_overlap_left_size_ = 0;
  int64_t right_rank_overlap_right_size_ = 0;
  int64_t w_dimension_shard_num_ = 1;
  Shape input_slice_shape_;

  bool left_need_send_ = false;
  bool left_need_recv_ = false;
  bool right_need_send_ = false;
  bool right_need_recv_ = false;
  Shape left_strided_slice_begin_;
  Shape left_strided_slice_end_;
  Shape left_strided_slice_strides_;
  Shape right_strided_slice_begin_;
  Shape right_strided_slice_end_;
  Shape right_strided_slice_strides_;

  std::vector<int64_t> send_rank_ids_;
  std::vector<int64_t> recv_rank_ids_;
  Shapes send_shapes_;
  Shapes recv_shapes_;

  GenerateGraph gen_g_ = GenerateGraph(attrs_);

  virtual Status CheckHWStrategy(int64_t h_strategy, int64_t w_strategy);
  virtual void InferNewPadList();
  virtual int64_t ComputeOverlapLeftSizeByRankBias(int64_t rank_bias);
  virtual int64_t ComputeOverlapRightSizeByRankBias(int64_t rank_bias);

 private:
  Status CheckHWStrategySameMode(int64_t h_strategy, int64_t w_strategy);
  Status CheckHWStrategyValidMode(int64_t h_strategy, int64_t w_strategy);
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
