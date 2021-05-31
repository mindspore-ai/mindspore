/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TENSORDOT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TENSORDOT_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
enum AxesType {
  INT_TYPE = 0,
  TUPLE_TYPE,
  TUPLE_TUPLE_TYPE,
};

class TensorDotInfo : public OperatorInfo {
 public:
  TensorDotInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorDotCost>()) {}
  ~TensorDotInfo() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  Status PrepareStrategy(int32_t stage_id, size_t dev_num, Dimensions combined_partitions, size_t input0_shape_size,
                         size_t input1_shape_size, StrategyPtr *sp);

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  std::shared_ptr<Strategys> GenerateBatchStrategies() override;
  void InferTensorMapAxesInt(const TensorMap &tensor_map_index);
  void InferTensorMapAxesTuple(size_t size, const TensorMap &input_a_tensor_map, const TensorMap &tensor_map_index);
  void ShowAxes();
  Shape origin_dev_matrix_shape_;

  AxesType axes_type_ = INT_TYPE;
  int32_t axes_int_ = 1;
  std::vector<int32_t> axes_tuple_;
  std::vector<std::vector<int32_t>> axes_tuple_tuple_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TENSORDOT_INFO_H_
