/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/tensor.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ReduceMethod : public OperatorInfo {
 public:
  ReduceMethod(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ReduceMethod() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  std::string reduce_method_;
  bool keepdims_ = false;
  bool cross_batch_ = false;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override;
  Dimensions InferOutputStrategy();
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferMirrorOps() override;
  virtual std::vector<int64_t> reduce_dim();
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
};

class ReduceMaxInfo : public ReduceMethod {
 public:
  ReduceMaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMaxCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  ~ReduceMaxInfo() override = default;
};

class ArgMaxWithValueInfo : public ReduceMethod {
 public:
  ArgMaxWithValueInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArgMaxWithValueCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  ~ArgMaxWithValueInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferAsLossDivisor() override;
};

class ArgMinWithValueInfo : public ArgMaxWithValueInfo {
 public:
  ArgMinWithValueInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : ArgMaxWithValueInfo(name, inputs_shape, outputs_shape, attrs) {
    reduce_method_ = REDUCE_OP_MIN;
  }

  ~ArgMinWithValueInfo() override = default;
};

class ReduceMeanInfo : public ReduceMethod {
 public:
  ReduceMeanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMeanCost>()) {}

  ~ReduceMeanInfo() override = default;

 protected:
  Status InferForwardCommunication() override;
};

class ReduceSumInfo : public ReduceMethod {
 public:
  ReduceSumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_SUM;
  }

  ~ReduceSumInfo() override = default;
};

class ReduceAnyInfo : public ReduceMethod {
 public:
  ReduceAnyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_ANY;
  }
  ~ReduceAnyInfo() override = default;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
};

class ReduceMinInfo : public ReduceMethod {
 public:
  ReduceMinInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMinCost>()) {
    reduce_method_ = REDUCE_OP_MIN;
  }

  ~ReduceMinInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_
