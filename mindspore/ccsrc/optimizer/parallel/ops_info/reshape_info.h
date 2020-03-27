/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_RESHAPE_INFO_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_RESHAPE_INFO_H_

#include <ir/value.h>

#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "optimizer/parallel/ops_info/operator_info.h"
#include "optimizer/parallel/strategy.h"

namespace mindspore {
namespace parallel {
/*
 * parallel class for Reshape Primitive
 */
class ReshapeInfo : public OperatorInfo {
 public:
  ReshapeInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
              const PrimitiveAttrs& attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs),
        dev_num_(0),
        input_layout_set_flag_(false),
        output_layout_set_flag_(false) {
    reshape_cost_ptr_ = std::make_shared<ReshapeCost>();
  }
  ~ReshapeInfo() override = default;
  Status Init(const StrategyPtr& strategy) override;
  void SetInputLayout(const TensorLayout& input_layout) {
    input_layout_ = input_layout;
    input_layout_set_flag_ = true;
  }
  void SetOutputLayout(const TensorLayout& output_layout) {
    output_layout_ = output_layout;
    output_layout_set_flag_ = true;
  }
  Status InitForCostModel(const StrategyPtr& strategy) override;
  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr& strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return reshape_cost_ptr_; }

 protected:
  Status CheckStrategy(const StrategyPtr& strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorLayout(TensorLayouts* inputs_layout, TensorLayouts* outputs_layout);
  Status GetAttrs() override;
  Strategys GetOutputsStrategy();
  ReshapeCostPtr reshape_cost_ptr_;

 private:
  Status GetParameterInput();
  Status ComputeReplaceOp();
  void InferTensorInfoByLayout();
  void device_number(const StrategyPtr& strategy);
  Status InferDefaultLayout(const Shape& shape, TensorLayout* const layout);

  int32_t dev_num_;
  std::vector<int32_t> parameter_input_v_;
  Dimensions input_strategy_;
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  bool input_layout_set_flag_;
  bool output_layout_set_flag_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_RESHAPE_INFO_H_
