/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_LAYER_NORM_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_LAYER_NORM_INFO_H_

#include <string>
#include <memory>
#include <vector>
#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t LAYER_NORM_INPUT_SIZE = 3;
constexpr size_t LAYER_NORM_INPUT_INDEX = 0;
constexpr size_t LAYER_NORM_GAMMA_INDEX = 1;
constexpr size_t LAYER_NORM_BETA_INDEX = 2;
constexpr char BEGIN_NORM_AXIS[] = "begin_norm_axis";

// The dimensions of input tensor starting from begin norm axis cannot be split. Other dimensions can be split
// arbitrarily. Gamma and beta should match input to meet the broadcast requirements of mul and add.
class LayerNormInfo : public OperatorInfo {
 public:
  LayerNormInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<LayerNormCost>()),
        begin_norm_axis_(0) {}
  ~LayerNormInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferAsLossDivisor() override;
  Status CreateInputTensorMap(size_t input_index);
  Status GenerateGammaAndBetaStrategies(const std::vector<StrategyPtr> &sp_vector);
  Status InitShapes();
  Status InferMirrorOps() override;
  Status InferOutputTensorInfo() override;
  Status InferForwardCommunicationByLayout() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;

 private:
  size_t begin_norm_axis_;
  Shape input_shape_;
  Shape gamma_shape_;
  Shape beta_shape_;
  Status InferOutputLayout();
  TensorLayout output_infer_tensor_layout_;
  TensorLayout mean_infer_tensor_layout_;
  TensorLayout var_infer_tensor_layout_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_LAYER_NORM_INFO_H_
