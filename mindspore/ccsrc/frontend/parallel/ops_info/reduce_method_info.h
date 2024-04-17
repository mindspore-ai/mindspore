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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/strategy.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace parallel {
class ReduceMethod : public OperatorInfo {
 public:
  ReduceMethod(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ReduceMethod() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  std::string reduce_method_;
  bool keepdims_ = false;
  bool cross_batch_ = false;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;

  Status InferMirrorOps() override;
  Status GetAttrs() override;
  virtual std::vector<int64_t> reduce_dim();
};

class ArgMaxWithValueInfo : public ReduceMethod {
 public:
  ArgMaxWithValueInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArgMaxWithValueCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  ~ArgMaxWithValueInfo() override = default;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferAsLossDivisor() override;
  Status GetAttrs() override;
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

class ArgmaxInfo : public ReduceMethod {
 public:
  ArgmaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArgmaxCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  ~ArgmaxInfo() override = default;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
};

class ArgminInfo : public ArgmaxInfo {
 public:
  ArgminInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArgmaxInfo(name, inputs_shape, outputs_shape, attrs) {
    reduce_method_ = REDUCE_OP_MIN;
  }

  ~ArgminInfo() override = default;
};

class SquareSumAllInfo : public ReduceMethod {
 public:
  SquareSumAllInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<SquareSumAllCost>()) {
    reduce_method_ = REDUCE_OP_SUM;
  }
  ~SquareSumAllInfo() override = default;

  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferMirrorOps() override;
  Status InferAsLossDivisor() override;

 private:
  Status InferGroup();
  Status ComputeReplaceGraph(const CNodePtr &cnode);

  std::vector<Group> group_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_SUM_INFO_H_
