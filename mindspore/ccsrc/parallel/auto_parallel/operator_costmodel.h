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

#ifndef PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_
#define PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_

#include <memory>
#include <vector>
#include "parallel/device_manager.h"
#include "parallel/tensor_layout/tensor_info.h"

namespace mindspore {
namespace parallel {
#define MAXIMUM_INPUT_NUMBER 100
#define DEFAULT_DATA_TYPE_LENGTH 4
#define DROPOUT_COST_RATE 1.125  // the DropoutGenMask need 12.5% memory
#define GATHERV2_COST_WEIGHT0 3
#define GATHERV2_COST_WEIGHT1 7
#define GATHERV2_COST_WEIGHT2 2
#define GATHERV2_COST_WEIGHT3 6

class OperatorCost;
using OperatorCostPtr = std::shared_ptr<OperatorCost>;

template <typename T>
double ListProduct(std::vector<T> vec) {
  double result = 1;
  for (size_t i = 0; i < vec.size(); ++i) {
    result *= vec[i];
  }
  return result;
}
// NOTE: Currently, the returned value in each method is bytes of memory size, which is calculated by the number of
// entries timing the length of each entry's data type
class OperatorCost {
 public:
  explicit OperatorCost(bool is_inputs_related) : inputs_related_(is_inputs_related) {
    // this is only for the case when set_is_parameter() and SetInputAndOutputTypeLength() are not invoked
    for (size_t i = 0; i < MAXIMUM_INPUT_NUMBER; ++i) {
      is_parameter_.push_back(false);
      is_parameter_involve_.push_back(false);
      inputs_type_lengths_.push_back(DEFAULT_DATA_TYPE_LENGTH);
      outputs_type_lengths_.push_back(DEFAULT_DATA_TYPE_LENGTH);
    }
  }
  OperatorCost() : inputs_related_(false) {
    // this is only for the case when set_is_parameter() and SetInputAndOutputTypeLength() are not invoked
    for (size_t i = 0; i < MAXIMUM_INPUT_NUMBER; ++i) {
      is_parameter_.push_back(false);
      is_parameter_involve_.push_back(false);
      inputs_type_lengths_.push_back(DEFAULT_DATA_TYPE_LENGTH);
      outputs_type_lengths_.push_back(DEFAULT_DATA_TYPE_LENGTH);
    }
  }
  virtual ~OperatorCost() = default;

  void set_is_parameter(const std::vector<bool> &is_parameter);
  void set_is_parameter_involve(const std::vector<bool> &);
  void set_output_parameter_involve(int);
  void SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths, const std::vector<size_t> &output_lengths);
  std::vector<size_t> inputs_type_lengths() const { return inputs_type_lengths_; }
  std::vector<size_t> outputs_type_lengths() const { return outputs_type_lengths_; }

  // per device communication cost
  virtual double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const = 0;
  virtual double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const = 0;
  virtual double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                     int32_t stage_id) const = 0;
  // per device computation cost
  virtual double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const = 0;
  virtual double GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                           const std::vector<TensorInfo> &outputs, int32_t stage_id) const = 0;
  virtual double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs,
                                            const std::vector<TensorInfo> &outputs, int32_t stage_id) const = 0;
  // per device PEAK memory cost in a training iteration
  // Typically, the PEAK memory cost contributed by an operator is its output (if the output is parameter-invovled),
  // plus necessary inputs.
  virtual double GetMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const;

 protected:
  // For each input in 'inputs_', a bool variable is true if the corresponding one is a parameter or a output of
  // pre-operator that has parameters as input.
  std::vector<bool> is_parameter_involve_;
  int output_parameter_involve_ = -1;  // -1: unset; 0: not parameter_involved; 1: parameter_involved
  // Whether the inputs are related or not? For example, TensorAdd's two inputs are independent (not related), while
  // Mul's two inputs are dependent (related).
  bool inputs_related_;
  // for each input in 'inputs_', there is a bool variable indicating whether that the corresponding input is parameter
  std::vector<bool> is_parameter_;
  // for each input and output, the followings record the number of bytes of each element
  std::vector<size_t> inputs_type_lengths_;
  std::vector<size_t> outputs_type_lengths_;
};

using OperatorCostPtr = std::shared_ptr<OperatorCost>;

class MatMulCost : public OperatorCost {
 public:
  explicit MatMulCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  MatMulCost() : OperatorCost(true) {}
  ~MatMulCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using MatMulCostPtr = std::shared_ptr<MatMulCost>;

class ActivationCost : public OperatorCost {
 public:
  explicit ActivationCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  ActivationCost() : OperatorCost(false) {}
  ~ActivationCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using ActivationCostPtr = std::shared_ptr<ActivationCost>;
using TransposeCost = ActivationCost;
using TransposeCostPtr = std::shared_ptr<TransposeCost>;

class SoftmaxCost : public OperatorCost {
 public:
  explicit SoftmaxCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  SoftmaxCost() : OperatorCost(false) {}
  ~SoftmaxCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t) const override;
};
using SoftmaxCostPtr = std::shared_ptr<SoftmaxCost>;

class TmpIdentityCost : public OperatorCost {
 public:
  explicit TmpIdentityCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  TmpIdentityCost() : OperatorCost(false) {}
  ~TmpIdentityCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
  // per device PEAK memory cost in a training iteration
  double GetMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const override;
};
using TmpIdentityCostPtr = std::shared_ptr<TmpIdentityCost>;

class BatchParallelCost : public OperatorCost {
 public:
  explicit BatchParallelCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  BatchParallelCost() : OperatorCost(false) {}
  ~BatchParallelCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using BatchParallelCostPtr = std::shared_ptr<BatchParallelCost>;

class VirtualDatasetCost : public OperatorCost {
 public:
  explicit VirtualDatasetCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  VirtualDatasetCost() : OperatorCost(false) {}
  ~VirtualDatasetCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int32_t) const override {
    return 0.0;
  }
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
  // per device PEAK memory cost in a training iteration
  double GetMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const override {
    return 0.0;
  }
};
using VirtualDatasetCostPtr = std::shared_ptr<VirtualDatasetCost>;

class GeneratorBaseCost : public OperatorCost {
 public:
  explicit GeneratorBaseCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  GeneratorBaseCost() : OperatorCost(false) {}
  ~GeneratorBaseCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  // Inputs vector is empty for generator ops.
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int32_t) const override {
    return 0.0;
  }
  // Generator ops don't have backward steps.
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
};
using GeneratorBaseCostPtr = std::shared_ptr<GeneratorBaseCost>;

class PReLUCost : public OperatorCost {
 public:
  explicit PReLUCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  PReLUCost() : OperatorCost(true) {}
  ~PReLUCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using PReLUCostPtr = std::shared_ptr<PReLUCost>;

class OneHotCost : public OperatorCost {
 public:
  explicit OneHotCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  OneHotCost() : OperatorCost(true) {}
  ~OneHotCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using OneHotCostPtr = std::shared_ptr<OneHotCost>;

class SoftmaxCrossEntropyWithLogitsCost : public OperatorCost {
 public:
  explicit SoftmaxCrossEntropyWithLogitsCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  SoftmaxCrossEntropyWithLogitsCost() : OperatorCost(false) {}
  ~SoftmaxCrossEntropyWithLogitsCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using SoftmaxCrossEntropyWithLogitsCostPtr = std::shared_ptr<SoftmaxCrossEntropyWithLogitsCost>;

class ReshapeCost : public OperatorCost {
 public:
  explicit ReshapeCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  ReshapeCost() : OperatorCost(true) {}

  ~ReshapeCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }

  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;

  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }

  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;

  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using ReshapeCostPtr = std::shared_ptr<ReshapeCost>;

class ArithmeticCost : public OperatorCost {
 public:
  explicit ArithmeticCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  ArithmeticCost() : OperatorCost(false) {}
  ~ArithmeticCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override;

  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t stage_id) const override;
};
using ArithmeticCostPtr = std::shared_ptr<ArithmeticCost>;
using BiasAddCost = ArithmeticCost;
using BiasAddCostPtr = std::shared_ptr<BiasAddCost>;

class ReduceMethodCost : public OperatorCost {
 public:
  explicit ReduceMethodCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  ReduceMethodCost() : OperatorCost(true) {}
  ~ReduceMethodCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
  void set_cross_batch(bool cb) { cross_batch_ = cb; }

 protected:
  bool cross_batch_ = false;
};
using ReduceMethodCostPtr = std::shared_ptr<ReduceMethodCost>;

class ReduceMeanCost : public ReduceMethodCost {
 public:
  explicit ReduceMeanCost(bool is_inputs_related) : ReduceMethodCost(is_inputs_related) {}
  ReduceMeanCost() : ReduceMethodCost(true) {}
  ~ReduceMeanCost() override = default;

  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
};
using ReduceMeanCostPtr = std::shared_ptr<ReduceMeanCost>;

class GetNextCost : public OperatorCost {
 public:
  explicit GetNextCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  GetNextCost() : OperatorCost(false) {}
  ~GetNextCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  // Inputs vector is empty for generator ops.
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int32_t) const override {
    return 0.0;
  }
  // Generator ops don't have backward steps.
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
};
using GetNextCostPtr = std::shared_ptr<GetNextCost>;

class DropOutCost : public OperatorCost {
 public:
  explicit DropOutCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  DropOutCost() : OperatorCost(true) {}
  ~DropOutCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int32_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
};

using DropOutCostPtr = std::shared_ptr<DropOutCost>;

class LayerNormCost : public OperatorCost {
 public:
  explicit LayerNormCost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  LayerNormCost() : OperatorCost(true) {}
  ~LayerNormCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int32_t) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int32_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int32_t) const override {
    return 0.0;
  }
};

using DropOutCostPtr = std::shared_ptr<DropOutCost>;

class GatherV2Cost : public OperatorCost {
 public:
  explicit GatherV2Cost(bool is_inputs_related) : OperatorCost(is_inputs_related) {}
  GatherV2Cost() : OperatorCost(true) {}
  ~GatherV2Cost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t) const override;
};

using GatherV2CostPtr = std::shared_ptr<GatherV2Cost>;

class GatherV2PCost : public OperatorCost {
 public:
  explicit GatherV2PCost(bool is_inputs_related) : OperatorCost(is_inputs_related), axis_(0) {}
  GatherV2PCost() : OperatorCost(true), axis_(0) {}
  ~GatherV2PCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int32_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int32_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int32_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int32_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int32_t) const override;
  void set_axis(int32_t axis) { axis_ = axis; }
  void set_strategy(const Shape &strategy) { strategy_ = strategy; }

 protected:
  int32_t axis_;
  Shape strategy_;
};

using GatherV2PCostPtr = std::shared_ptr<GatherV2PCost>;
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_
