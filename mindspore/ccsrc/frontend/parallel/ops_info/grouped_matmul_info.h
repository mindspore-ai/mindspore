/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GROUPED_MATMUL_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GROUPED_MATMUL_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
/*
 * parallel class for GroupedMatmul Primitive
 */
class GroupedMatmulInfo : public OperatorInfo {
 public:
  GroupedMatmulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<ActivationInfoCost>()) {}
  ~GroupedMatmulInfo() override = default;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override { return {}; }
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }

 protected:
  Status GetAttrs() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferForwardCommunication() override;
  TensorInfoBasePtr CreateTensorInfo(const Shape &device_matrix, const ShapeBasePtr &inputs_shape,
                                     const ShapeBasePtr &inputs_tensor_map);
  Status InferAsLossDivisor();
  void SetOptionalInputTensorMap(const size_t &index, size_t *valid_input_index);

 private:
  size_t split_item_ = 3;
  size_t mat_x_dimension_ = 0;
  size_t mat_w_dimension_ = 0;
  Shape origin_dev_matrix_shape_;
};
using GroupedMatmulInfoPtr = std::shared_ptr<GroupedMatmulInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GROUPED_MATMUL_INFO_H_
