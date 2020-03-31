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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_

#include <ir/value.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "parallel/ops_info/operator_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ActivationBase : public OperatorInfo {
 public:
  ActivationBase(const std::string& operator_name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                 const PrimitiveAttrs& attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~ActivationBase() override = default;

  Status Init(const StrategyPtr& strategy) override;
  Status InitForCostModel(const StrategyPtr& strategy) override;

 protected:
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
};

class Activation : public ActivationBase {
 public:
  Activation(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
             const PrimitiveAttrs& attrs)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs) {
    ac_cost_ptr_ = std::make_shared<ActivationCost>();
  }
  ~Activation() override = default;
  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr& strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return ac_cost_ptr_; }

 protected:
  Status CheckStrategy(const StrategyPtr& strategy) override;

 private:
  ActivationCostPtr ac_cost_ptr_;
};

class ActivationInfo : public Activation {
 public:
  ActivationInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                 const PrimitiveAttrs& attrs)
      : Activation(name, inputs_shape, outputs_shape, attrs) {}
  ~ActivationInfo() override = default;

 protected:
  Status GetAttrs() override;  // activation_type: relu, relu6, sigmoid
};

class ActivationOther : public Activation {
 public:
  ActivationOther(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                  const PrimitiveAttrs& attrs)
      : Activation(name, inputs_shape, outputs_shape, attrs) {}
  ~ActivationOther() override = default;

 protected:
  Status GetAttrs() override;
};

class GeluInfo : public ActivationOther {
 public:
  GeluInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~GeluInfo() override = default;
};

class TanhInfo : public ActivationOther {
 public:
  TanhInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~TanhInfo() override = default;
};

class Softmax : public ActivationBase {
 public:
  explicit Softmax(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                   const PrimitiveAttrs& attrs)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs) {
    sm_cost_ptr_ = std::make_shared<SoftmaxCost>();
  }
  ~Softmax() override = default;
  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr& strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return sm_cost_ptr_; }

 protected:
  Status CheckStrategy(const StrategyPtr& strategy) override;
  Status GetAttrs() override;

 private:
  std::vector<int32_t> axis_;
  SoftmaxCostPtr sm_cost_ptr_;
};

class SoftmaxInfo : public Softmax {
 public:
  SoftmaxInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
              const PrimitiveAttrs& attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~SoftmaxInfo() override = default;
};

class LogSoftmaxInfo : public Softmax {
 public:
  LogSoftmaxInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                 const PrimitiveAttrs& attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~LogSoftmaxInfo() override = default;
};

class ReLUInfo : public ActivationOther {
 public:
  ReLUInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~ReLUInfo() override = default;
};

class CastInfo : public ActivationOther {
 public:
  CastInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~CastInfo() override = default;

 protected:
  Status InferMirrorOps() override;
};

class SqrtInfo : public ActivationOther {
 public:
  SqrtInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~SqrtInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPS_INFO_PARALLEL_ACTIVATION_INFO_H_
