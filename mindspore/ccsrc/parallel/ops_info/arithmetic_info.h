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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "ir/value.h"
#include "parallel/ops_info/operator_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ArithmeticBase : public OperatorInfo {
 public:
  ArithmeticBase(const std::string& operator_name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                 const PrimitiveAttrs& attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs) {
    arithmeticcost_ptr_ = std::make_shared<ArithmeticCost>();
  }
  ~ArithmeticBase() override = default;
  Status Init(const StrategyPtr& strategy) override;
  Status InitForCostModel(const StrategyPtr& strategy) override;
  Status GenerateStrategies(int32_t) override;
  Status SetCostUnderStrategy(const StrategyPtr&) override;
  OperatorCostPtr GetOperatorCost() const override { return arithmeticcost_ptr_; }
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr& strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts* inputs_layout, TensorLayouts* outputs_layout, const Shape& dev_matrix_array);
  Shapes InferExpendShape();
  ArithmeticCostPtr arithmeticcost_ptr_;
};

class SubInfo : public ArithmeticBase {
 public:
  SubInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~SubInfo() override = default;
};

class TensorAddInfo : public ArithmeticBase {
 public:
  TensorAddInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~TensorAddInfo() override = default;
};

class MulInfo : public ArithmeticBase {
 public:
  MulInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~MulInfo() override = default;
};

class DivInfo : public ArithmeticBase {
 public:
  DivInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~DivInfo() override = default;
};

class RealDivInfo : public ArithmeticBase {
 public:
  RealDivInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
              const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~RealDivInfo() override = default;
};

class FloorDivInfo : public ArithmeticBase {
 public:
  FloorDivInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
               const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~FloorDivInfo() override = default;
};

class GreaterInfo : public ArithmeticBase {
 public:
  GreaterInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
              const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~GreaterInfo() override = default;
};

class AssignSubInfo : public ArithmeticBase {
 public:
  AssignSubInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~AssignSubInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPS_INFO_PARALLEL_ARITHMETIC_INFO_H_
