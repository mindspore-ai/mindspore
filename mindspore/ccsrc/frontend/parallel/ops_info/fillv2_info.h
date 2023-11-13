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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FILLV2_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FILLV2_INFO_H_

#include <memory>
#include <vector>
#include <string>

#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
class FillV2Info : public OperatorInfo {
 public:
  FillV2Info(const std::string &name, const Shapes &input_shape, const Shapes &output_shape,
             const PrimitiveAttrs &attrs)
      : OperatorInfo(name, input_shape, output_shape, attrs, std::make_shared<FillV2Cost>()) {}
  ~FillV2Info() = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status InferAttrs() override;
  Status GetAttrs() override { return SUCCESS; };
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; };

 private:
  void ResetInputsShape();
  void ReplaceDynamicInput(const CNodePtr &cnode, const Shape &strategy);
  Shape GetShapeFromTensor(const tensor::TensorPtr &shape_tensor);
  Shapes fake_inputs_shape_;  // if dynamic shape, replace -1 to 1
  bool is_dynamic_shape_ = false;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FILLV2_INFO_H_
