/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESHAPE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESHAPE_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
/*
 * parallel class for Reshape Primitive
 */
class ReshapeInfo : public OperatorInfo {
 public:
  ReshapeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReshapeCost>()),
        dev_num_(0),
        pre_operator_index_(0),
        next_operator_index_(0),
        input_layout_set_flag_(false),
        output_layout_set_flag_(false) {}
  ~ReshapeInfo() override = default;
  Status Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
              const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts = {},
              const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts = {}) override;
  void SetInputLayout(const TensorLayout &input_layout) {
    input_layout_ = input_layout;
    if (input_layout_.fine_grain_block_index() >= 0) {
      auto cnode_primitive = GetCNodePrimitive(cnode_);
      MS_EXCEPTION_IF_NULL(cnode_primitive);
      cnode_primitive->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                               MakeValue<int64_t>(input_layout_.fine_grain_block_index()));
    }
    input_layout_set_flag_ = true;
  }
  void SetOutputLayout(const TensorLayout &output_layout) {
    output_layout_ = output_layout;
    if (output_layout_.fine_grain_block_index() >= 0) {
      auto cnode_primitive = GetCNodePrimitive(cnode_);
      MS_EXCEPTION_IF_NULL(cnode_primitive);
      cnode_primitive->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                               MakeValue<int64_t>(output_layout_.fine_grain_block_index()));
    }
    output_layout_set_flag_ = true;
  }
  void SetCostForReshape(const mindspore::parallel::StrategyPtr &strategy);
  void SetCostForReshapeWithParameter();
  void set_pre_operator_name(const std::string &pre_name) { pre_operator_name_ = pre_name; }
  void set_next_operator_name(const std::string &next_name) { next_operator_name_ = next_name; }
  void set_pre_operator_index(int64_t pre_index) { pre_operator_index_ = pre_index; }
  void set_next_operator_index(int64_t next_index) { next_operator_index_ = next_index; }
  StrategyPtr get_input_shard_strategy();
  Status GenerateStrategyCosts(
    const std::vector<std::shared_ptr<StrategyWithCost>> &pre_stra_costs,
    std::vector<std::pair<std::vector<std::shared_ptr<StrategyWithCost>>, int64_t>> next_costs_index, int64_t out_index,
    bool is_prev_param, bool is_next_reshape);
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  std::string pre_operator_name() const { return pre_operator_name_; }
  std::string next_operator_name() const { return next_operator_name_; }
  int64_t pre_operator_index() const { return pre_operator_index_; }
  int64_t next_operator_index() const { return next_operator_index_; }

  int64_t GetSWCIndexByOutputLayoutWithZeroComm(const TensorLayout &output_layout);
  int64_t GetSWCIndexByOutputLayoutWithMiniComm(const TensorLayout &output_layout);
  int64_t GetSWCIndexByInputLayoutWithZeroComm(const TensorLayout &input_layout);
  int64_t GetSWCIndexByInputLayoutWithMiniComm(const TensorLayout &input_layout);
  bool CheckStrategyConsistencyByOutputLayout(int64_t swc_index, const TensorLayout &output_layout) const;
  bool CheckStrategyConsistencyByInputLayout(int64_t swc_index, const TensorLayout &input_layout) const;

  TensorLayout GetInputLayoutBySWCIndex(int64_t swc_index) const;
  TensorLayout GetOutputLayoutBySWCIndex(int64_t swc_index) const;

  bool InterleavedParallel() const { return interleaved_parallel_; }
  TensorRedistributionPtr ReshapeRedistribution();

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout);
  Status GetAttrs() override { return SUCCESS; }

 private:
  void SkipReshapeRedistribution();
  Status SetReshapeRedistribution(bool *has_replaced);
  Status ComputeReplaceOp();
  Status ComputeReplaceOpForDynamicShape();
  void InferTensorInfoByLayout();
  void device_number();
  Status InferDefaultLayout(const Shape &shape, TensorLayout *const layout);
  std::vector<int64_t> GetInputShape(const AnfNodePtr &shape_input_node);
  void ChangeDynamicDstShapeForSkipRedistribution(const AnfNodePtr &shape_input_node);
  void ChangeDstShape();

  int64_t dev_num_;
  int64_t pre_operator_index_;
  int64_t next_operator_index_;
  std::vector<int64_t> parameter_input_v_;
  std::vector<StrategyPtr> sp_vector_;
  Dimensions input_strategy_;
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  bool input_layout_set_flag_;
  bool output_layout_set_flag_;
  bool is_generating_costs_ = false;
  bool is_skip_ = false;
  bool interleaved_parallel_ = false;
  std::string pre_operator_name_;
  std::string next_operator_name_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RESHAPE_INFO_H_
