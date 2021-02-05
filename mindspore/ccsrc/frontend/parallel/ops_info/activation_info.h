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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_

#include <ir/value.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ActivationBase : public OperatorInfo {
 public:
  ActivationBase(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ActivationBase() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

 protected:
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
};

class Activation : public ActivationBase {
 public:
  Activation(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~Activation() override = default;
  Status GenerateStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
};

class ActivationInfo : public Activation {
 public:
  ActivationInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : Activation(name, inputs_shape, outputs_shape, attrs, std::make_shared<ActivationInfoCost>()) {}
  ~ActivationInfo() override = default;

 protected:
  Status GetAttrs() override;  // activation_type: relu, relu6, sigmoid
};

class ActivationOther : public Activation {
 public:
  ActivationOther(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : Activation(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ActivationOther() override = default;

 protected:
  Status GetAttrs() override;
};

class GeLUInfo : public ActivationOther {
 public:
  GeLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<GeLUCost>()) {}
  ~GeLUInfo() override = default;
};

class TanhInfo : public ActivationOther {
 public:
  TanhInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<TanhCost>()) {}
  ~TanhInfo() override = default;
};

class Softmax : public ActivationBase {
 public:
  explicit Softmax(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ActivationBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftmaxCost>()) {}
  ~Softmax() override = default;
  Status GenerateStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override;

 private:
  std::vector<int64_t> axis_;
};

class SoftmaxInfo : public Softmax {
 public:
  SoftmaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~SoftmaxInfo() override = default;
};

class LogSoftmaxInfo : public Softmax {
 public:
  LogSoftmaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : Softmax(name, inputs_shape, outputs_shape, attrs) {}
  ~LogSoftmaxInfo() override = default;
};

class EluInfo : public ActivationOther {
 public:
  EluInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<EluCost>()) {}
  ~EluInfo() override = default;
};

class ReLUInfo : public ActivationOther {
 public:
  ReLUInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLUCost>()) {}
  ~ReLUInfo() override = default;
};

class RepeatElementsInfo : public ActivationOther {
 public:
  RepeatElementsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RepeatElementsCost>()) {}
  ~RepeatElementsInfo() override = default;
};

class ReLU6Info : public ActivationOther {
 public:
  ReLU6Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReLU6Cost>()) {}
  ~ReLU6Info() override = default;
};

class SoftsignInfo : public ActivationOther {
 public:
  SoftsignInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftsignCost>()) {}
  ~SoftsignInfo() override = default;
};

class SoftplusInfo : public ActivationOther {
 public:
  SoftplusInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SoftplusCost>()) {}
  ~SoftplusInfo() override = default;
};

class CastInfo : public ActivationOther {
 public:
  CastInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<CastCost>()) {}
  ~CastInfo() override = default;

 protected:
  Status InferMirrorOps() override;
};

class SqrtInfo : public ActivationOther {
 public:
  SqrtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SqrtCost>()) {}
  ~SqrtInfo() override = default;
};

class NegInfo : public ActivationOther {
 public:
  NegInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<NegCost>()) {}
  ~NegInfo() override = default;
};

class ExpandDimsInfo : public ActivationOther {
 public:
  ExpandDimsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ExpandDimsCost>()) {}
  ~ExpandDimsInfo() override = default;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferMirrorOps() override;
  Status InferTensorStrategy();

 private:
  int64_t positive_axis_ = -1;
  Strategys inputs_strategy_;
  Strategys outputs_strategy_;
};

class SqueezeInfo : public ActivationOther {
 public:
  SqueezeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SqueezeCost>()) {}
  ~SqueezeInfo() override = default;

 protected:
  Status InferAxis(const ValueTuplePtr &value_tuple);
  Status GetAttrs() override;
  Status InferReplaceOps(const StrategyPtr &strategy);
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status Init(const StrategyPtr &strategy) override;

 private:
  ValueTuplePtr axis_;
};

class SquareInfo : public ActivationOther {
 public:
  SquareInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SquareCost>()) {}
  ~SquareInfo() override = default;
};

class SigmoidInfo : public ActivationOther {
 public:
  SigmoidInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SigmoidCost>()) {}
  ~SigmoidInfo() override = default;
};

class DropoutInfo : public ActivationOther {
 public:
  DropoutInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<DropOutCost>()) {}
  ~DropoutInfo() override = default;
  Status GenerateStrategies(int64_t stage_id) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override { return SUCCESS; }
  Status InferTensorInfo() override;

 private:
  bool IsRepeatedStrategy(const StrategyPtr &sp);
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ACTIVATION_INFO_H_
