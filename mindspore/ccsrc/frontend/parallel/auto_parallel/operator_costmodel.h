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
#ifndef PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_
#define PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_

#include <memory>
#include <vector>
#include <map>
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/auto_parallel/costmodel.h"

namespace mindspore {
namespace parallel {
constexpr size_t MAXIMUM_INPUT_NUMBER = 100;
constexpr size_t DEFAULT_DATA_TYPE_LENGTH = 4;
constexpr double DROPOUT_COST_RATE = 1.125;  // the DropoutGenMask need 12.5% memory
constexpr size_t GATHERV2_COST_WEIGHT0 = 3;
constexpr size_t GATHERV2_COST_WEIGHT1 = 7;
constexpr size_t GATHERV2_COST_WEIGHT2 = 2;
constexpr size_t GATHERV2_COST_WEIGHT3 = 6;

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
  OperatorCost() {
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
  void set_output_parameter_involve(int64_t);
  void set_output_critical(int64_t);
  void SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths, const std::vector<size_t> &output_lengths);
  std::vector<size_t> inputs_type_lengths() const { return inputs_type_lengths_; }
  std::vector<size_t> outputs_type_lengths() const { return outputs_type_lengths_; }

  // per device communication cost
  virtual double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const = 0;
  virtual double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const = 0;
  virtual double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                     int64_t stage_id) const = 0;
  // per device computation cost
  virtual double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const = 0;
  virtual double GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                           const std::vector<TensorInfo> &outputs, int64_t stage_id) const = 0;
  virtual double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs,
                                            const std::vector<TensorInfo> &outputs, int64_t stage_id) const = 0;
  virtual void CalculateOutputInMemory() = 0;
  virtual void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) = 0;
  bool is_output_in_memory() const { return is_output_should_in_memory_; }
  // per device PEAK memory cost in a training iteration
  // Typically, the PEAK memory cost contributed by an operator is its output (if the output is parameter-involved),
  // plus necessary inputs.
  virtual double GetMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const;
  // Contributing the input part for 'GetMemoryCost'
  double GetInputMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const;
  // Contributing the output part for 'GetMemoryCost'
  double GetOutputMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs) const;
  // per device memory cost in a inference phase
  double GetMemoryCostForInference(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &outputs) const;

 protected:
  // For each input in 'inputs_', a bool variable is true if the corresponding one is a parameter or a output of
  // pre-operator that has parameters as input.
  std::vector<bool> is_parameter_involve_;
  int64_t output_parameter_involve_ = -1;  // -1: unset; 0: not parameter_involved; 1: parameter_involved
  // For each input in 'inputs_', there is a bool variable indicating whether that the corresponding input is parameter
  std::vector<bool> is_parameter_;
  // Whether the input should keep in memory in training phase. It depends on the operator and the operator's
  // previous operators.
  std::vector<bool> is_inputs_should_in_memory_;
  // Whether the output should keep in memory in training phase. It depends on 'is_parameter_involve_' and the operator.
  bool is_output_should_in_memory_ = false;
  // For each input and output, the followings record the number of bytes of each element
  std::vector<size_t> inputs_type_lengths_;
  std::vector<size_t> outputs_type_lengths_;
  // Whether the output is critical, which means that this output is included in calculating peak memory cost
  // in the inference phase.
  int64_t is_outputs_critical_ = -1;
};

class MatMulCost : public OperatorCost {
 public:
  MatMulCost() : OperatorCost() {}
  ~MatMulCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  void CalculateOutputInMemory() override;
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using TensorDotCost = MatMulCost;

class BatchNormCost : public OperatorCost {
 public:
  BatchNormCost() : OperatorCost() {}
  ~BatchNormCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override;
  void CalculateOutputInMemory() override;
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class CastCost : public OperatorCost {
 public:
  CastCost() : OperatorCost() {}
  ~CastCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using RepeatElementsCost = CastCost;
using NegCost = CastCost;
using ExpandDimsCost = CastCost;
using SqueezeCost = CastCost;
using ConcatCost = CastCost;
using LogicalNotCost = CastCost;
using SignCost = CastCost;
using FloorCost = CastCost;
using RoundCost = CastCost;
using CeilCost = CastCost;
using ZerosLikeCost = CastCost;
using OnesLikeCost = CastCost;
using RangeCost = CastCost;
using SplitCost = CastCost;
using ScatterUpdateCost = CastCost;
using RandomDistributeCost = CastCost;
using FillV2Cost = CastCost;
using ResizeBilinearCost = CastCost;
using BoundingBoxEncodeCost = CastCost;
using IOUCost = CastCost;
using RandomChoicWithMaskCost = CastCost;
using IsFiniteCost = CastCost;
using RintCost = CastCost;
using GammaCost = CastCost;
using LinSpaceCost = CastCost;

class SqrtCost : public CastCost {
 public:
  SqrtCost() : CastCost() {}
  ~SqrtCost() override = default;
  // Taking account of output, not taking accounting of input
  void CalculateOutputInMemory() override;
};
using TanhCost = SqrtCost;
using EluCost = SqrtCost;
using ReLUCost = SqrtCost;
using SiLUCost = SqrtCost;
using identityCost = SqrtCost;
using SigmoidCost = SqrtCost;
using ReciprocalCost =
  SqrtCost;  // The derivative of 'Reciprocal' is different on 'Ascend' and 'GPU'. Here, 'Ascend' is chosen
using InvCost = SqrtCost;
using RsqrtCost = SqrtCost;
using AsinhCost = SqrtCost;
using AcoshCost = SqrtCost;
using TopKCost = SqrtCost;
using HShrinkCost = SqrtCost;
using HSigmoidCost = SqrtCost;
using MishCost = SqrtCost;
using SeLUCost = SqrtCost;
using SoftShrinkCost = SqrtCost;

class ReLU6Cost : public CastCost {
 public:
  ReLU6Cost() : CastCost() {}
  ~ReLU6Cost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using SoftsignCost = ReLU6Cost;
using SoftplusCost = ReLU6Cost;
using SquareCost = ReLU6Cost;
using ExpCost = ReLU6Cost;
using LogCost = ReLU6Cost;
using CosCost = ReLU6Cost;
using ACosCost = ReLU6Cost;
using AbsCost = ReLU6Cost;
using TanCost = ReLU6Cost;
using SinCost = ReLU6Cost;
using SinhCost = ReLU6Cost;
using Log1pCost = ReLU6Cost;
using Expm1Cost = ReLU6Cost;
using CoshCost = ReLU6Cost;
using AtanhCost = ReLU6Cost;
using AtanCost = ReLU6Cost;
using AsinCost = ReLU6Cost;
using ErfCost = ReLU6Cost;
using ErfcCost = ReLU6Cost;
using ActivationInfoCost = ReLU6Cost;
using SelectCost = ReLU6Cost;
using XlogyCost = ReLU6Cost;
using ErfinvCost = ReLU6Cost;

class TransposeCost : public CastCost {
 public:
  TransposeCost() : CastCost() {}
  ~TransposeCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class GeLUCost : public SqrtCost {
 public:
  GeLUCost() : SqrtCost() {}
  ~GeLUCost() override = default;
  // Taking account of input and output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using FastGeLUCost = GeLUCost;
using BesselI0eCost = GeLUCost;
using BesselI1eCost = GeLUCost;
using L2NormalizeCost = GeLUCost;
using MaxPoolCost = GeLUCost;

class SoftmaxCost : public OperatorCost {
 public:
  SoftmaxCost() : OperatorCost() {}
  ~SoftmaxCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t) const override;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

using CumSumCost = SoftmaxCost;
using CumProdCost = SoftmaxCost;

class TileCost : public SoftmaxCost {
 public:
  TileCost() : SoftmaxCost() {}
  ~TileCost() override = default;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class PackCost : public SoftmaxCost {
 public:
  PackCost() : SoftmaxCost() {}
  ~PackCost() override = default;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class BroadcastToCost : public SoftmaxCost {
 public:
  BroadcastToCost() : SoftmaxCost() {}
  ~BroadcastToCost() override = default;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class TmpIdentityCost : public OperatorCost {
 public:
  TmpIdentityCost() : OperatorCost() {}
  ~TmpIdentityCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using TmpIdentityCostPtr = std::shared_ptr<TmpIdentityCost>;

class BatchParallelCost : public OperatorCost {
 public:
  BatchParallelCost() : OperatorCost() {}
  ~BatchParallelCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class SparseSoftmaxCrossEntropyWithLogitsCost : public BatchParallelCost {
 public:
  SparseSoftmaxCrossEntropyWithLogitsCost() : BatchParallelCost() {}
  ~SparseSoftmaxCrossEntropyWithLogitsCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class VirtualDatasetCost : public OperatorCost {
 public:
  VirtualDatasetCost() : OperatorCost() {}
  ~VirtualDatasetCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override {
    return 0.0;
  }
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class GeneratorBaseCost : public OperatorCost {
 public:
  GeneratorBaseCost() : OperatorCost() {}
  ~GeneratorBaseCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  // Inputs vector is empty for generator ops.
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override {
    return 0.0;
  }
  // Generator ops don't have backward steps.
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
};
using GeneratorBaseCostPtr = std::shared_ptr<GeneratorBaseCost>;

class PReLUCost : public OperatorCost {
 public:
  PReLUCost() : OperatorCost() {}
  ~PReLUCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using PReLUCostPtr = std::shared_ptr<PReLUCost>;

class OneHotCost : public OperatorCost {
 public:
  OneHotCost() : OperatorCost() {}
  ~OneHotCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using OneHotCostPtr = std::shared_ptr<OneHotCost>;

class SoftmaxCrossEntropyWithLogitsCost : public OperatorCost {
 public:
  SoftmaxCrossEntropyWithLogitsCost() : OperatorCost() {}
  ~SoftmaxCrossEntropyWithLogitsCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class ReshapeCost : public OperatorCost {
 public:
  ReshapeCost() : OperatorCost() {}

  ~ReshapeCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }

  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;

  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }

  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;

  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using ReshapeCostPtr = std::shared_ptr<ReshapeCost>;

class SubCost : public OperatorCost {
 public:
  SubCost() : OperatorCost() {}
  ~SubCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                             int64_t stage_id) const override;

  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using TensorAddCost = SubCost;
using FloorDivCost = SubCost;
using AssignSubCost = SubCost;
using AssignAddCost = SubCost;
using LogicalAndCost = SubCost;
using LogicalOrCost = SubCost;
using BiasAddCost = SubCost;
using EqualCost = SubCost;
using ApproximateEqualCost = SubCost;
using NotEqualCost = SubCost;
using GreaterCost = SubCost;
using GreaterEqualCost = SubCost;
using LessCost = SubCost;
using LessEqualCost = SubCost;
using GatherNdCost = SubCost;
using BitwiseAndCost = SubCost;
using BitwiseOrCost = SubCost;
using BitwiseXorCost = SubCost;
using AddNCost = SubCost;
using InplaceAddCost = SubCost;
using InplaceSubCost = InplaceAddCost;
using InplaceUpdateCost = InplaceAddCost;
using MaskedFillCost = SubCost;

class MulCost : public SubCost {
 public:
  MulCost() : SubCost() {}
  ~MulCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

using MulNoNanCost = MulCost;
using GatherDCost = MulCost;
using LerpCost = MulCost;
using SquaredDifferenceCost = MulCost;

class DivCost : public SubCost {
 public:
  DivCost() : SubCost() {}
  ~DivCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using ReadDivCost = DivCost;
using TruncateDivCost = DivCost;
using XdivyCost = DivCost;
using CdistCost = DivCost;

class ModCost : public SubCost {
 public:
  ModCost() : SubCost() {}
  ~ModCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using FloorModCost = ModCost;
using TruncateModCost = ModCost;

class PowCost : public SubCost {
 public:
  PowCost() : SubCost() {}
  ~PowCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class AssignCost : public SubCost {
 public:
  AssignCost() : SubCost() {}
  ~AssignCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class SigmoidCrossEntropyWithLogitsCost : public SubCost {
 public:
  SigmoidCrossEntropyWithLogitsCost() : SubCost() {}
  ~SigmoidCrossEntropyWithLogitsCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class Atan2Cost : public SubCost {
 public:
  Atan2Cost() : SubCost() {}
  ~Atan2Cost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class DivNoNanCost : public SubCost {
 public:
  DivNoNanCost() : SubCost() {}
  ~DivNoNanCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class MaximumCost : public SubCost {
 public:
  MaximumCost() : SubCost() {}
  ~MaximumCost() override = default;
  // Taking account of input, not taking account of output
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using MinimumCost = MaximumCost;
using CumminCost = MaximumCost;

class SliceCost : public CastCost {
 public:
  SliceCost() : CastCost() {}
  ~SliceCost() override = default;
  // Not taking account of output, taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class StridedSliceCost : public CastCost {
 public:
  StridedSliceCost() : CastCost() {}
  ~StridedSliceCost() override = default;
  // Not taking account of output, taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class ReduceSumCost : public OperatorCost {
 public:
  ReduceSumCost() : OperatorCost() {}
  ~ReduceSumCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  void set_cross_batch(bool cb) { cross_batch_ = cb; }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;

 protected:
  bool cross_batch_ = false;
};
using ReduceMethodCost = ReduceSumCost;
using ReduceProdCost = ReduceSumCost;
using SquareSumAllCost = ReduceSumCost;
using L2LossCost = ReduceSumCost;
using KLDivLossCost = ReduceSumCost;

class ReduceMeanCost : public ReduceSumCost {
 public:
  ReduceMeanCost() : ReduceSumCost() {}
  ~ReduceMeanCost() override = default;

  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
};

class ReduceMinCost : public ReduceSumCost {
 public:
  ReduceMinCost() : ReduceSumCost() {}
  ~ReduceMinCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using ReduceMaxCost = ReduceMinCost;

class ArgMaxWithValueCost : public ReduceSumCost {
 public:
  ArgMaxWithValueCost() : ReduceSumCost() {}
  ~ArgMaxWithValueCost() override = default;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using ArgMinWithValueCost = ArgMaxWithValueCost;
using ArgmaxCost = ArgMaxWithValueCost;

class GetNextCost : public OperatorCost {
 public:
  GetNextCost() : OperatorCost() {}
  ~GetNextCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                            int64_t stage_id) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                             int64_t stage_id) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  // Inputs vector is empty for generator ops.
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override {
    return 0.0;
  }
  // Generator ops don't have backward steps.
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using GetNextCostPtr = std::shared_ptr<GetNextCost>;

class DSDMatmulCost : public OperatorCost {
 public:
  DSDMatmulCost() : OperatorCost() {}
  ~DSDMatmulCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  // Inputs vector is empty for generator ops.
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  // Generator ops don't have backward steps.
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using DSDMatmulCostPtr = std::shared_ptr<DSDMatmulCost>;

// For memory cost, taking account of output, not taking account of input
class DropOutCost : public SqrtCost {
 public:
  DropOutCost() : SqrtCost() {}
  ~DropOutCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
};

class DropOutDoMaskCost : public DropOutCost {
 public:
  DropOutDoMaskCost() : DropOutCost() {}
  ~DropOutDoMaskCost() override = default;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class UnsortedSegmentSumCost : public OperatorCost {
 public:
  UnsortedSegmentSumCost() : OperatorCost() {}
  ~UnsortedSegmentSumCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using UnsortedSegmentProdCost = UnsortedSegmentSumCost;

class UnsortedSegmentMinCost : public OperatorCost {
 public:
  UnsortedSegmentMinCost() : OperatorCost() {}
  ~UnsortedSegmentMinCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using UnsortedSegmentMaxCost = UnsortedSegmentMinCost;

class LayerNormCost : public OperatorCost {
 public:
  LayerNormCost() : OperatorCost() {}
  ~LayerNormCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class UniqueCost : public OperatorCost {
 public:
  UniqueCost() : OperatorCost() {}
  ~UniqueCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  // Taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class UniformCandidateSamplerCost : public OperatorCost {
 public:
  UniformCandidateSamplerCost() : OperatorCost() {}
  ~UniformCandidateSamplerCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class GatherV2Cost : public OperatorCost {
 public:
  GatherV2Cost() : OperatorCost() {}
  ~GatherV2Cost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t) const override;
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};

class GatherCost : public GatherV2Cost {
 public:
  GatherCost() : GatherV2Cost(), axis_(0) {}
  ~GatherCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t) const override;
  void set_axis(int64_t axis) { axis_ = axis; }
  void set_strategy(const Shape &strategy) { strategy_ = strategy; }

 protected:
  int64_t axis_;
  Shape strategy_;
};

class MatmulDDSCost : public OperatorCost {
 public:
  MatmulDDSCost() : OperatorCost() {}
  ~MatmulDDSCost() override = default;

  // per device communication cost
  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  };
  double GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const override {
    return 0.0;
  };

  // per device computation cost
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override {
    return 0.0;
  };
  // Not taking account of output
  void CalculateOutputInMemory() override;
  // Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
};
using MatmulDDSCostPtr = std::shared_ptr<MatmulDDSCost>;

class ScatterMathOpsCost : public OperatorCost {
 public:
  ScatterMathOpsCost() : OperatorCost() {}
  ~ScatterMathOpsCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return 0.0;
  }
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override {
    return 0.0;
  }
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                   int64_t stage_id) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override {
    return 0.0;
  }
  // Not taking account of output
  void CalculateOutputInMemory() override { is_output_should_in_memory_ = false; }
  // Not Taking account of input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override {
    is_inputs_should_in_memory_[0] = true;
  }

  void set_is_split_axis(bool is_split_axis) { is_split_axis_ = is_split_axis; }
  void set_coefficient(int32_t input_coefficient, int32_t indices_coefficient, int32_t updates_coefficient) {
    input_coefficient_ = input_coefficient;
    indices_coefficient_ = indices_coefficient;
    updates_coefficient_ = updates_coefficient;
  }

 protected:
  bool is_split_axis_ = false;
  int32_t input_coefficient_ = 1;
  int32_t indices_coefficient_ = 3;
  int32_t updates_coefficient_ = 2;
};

class ScatterNdOpsCost : public ScatterMathOpsCost {
 public:
  ScatterNdOpsCost() : ScatterMathOpsCost() {}
  ~ScatterNdOpsCost() override = default;
};

class TensorScatterOpsCost : public ScatterMathOpsCost {
 public:
  TensorScatterOpsCost() : ScatterMathOpsCost() {}
  ~TensorScatterOpsCost() override = default;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                    int64_t stage_id) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
};

class CropAndResizeCost : public OperatorCost {
 public:
  CropAndResizeCost() : OperatorCost() {}
  ~CropAndResizeCost() override = default;

  double GetCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                     int64_t stage_id) const override {
    return GetForwardCommCost(inputs, outputs, stage_id) + GetBackwardCommCost(inputs, outputs, stage_id);
  }
  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t) const override;
  double GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                             int64_t stage_id) const override;
  double GetComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t stage_id) const override {
    return GetForwardComputationCost(inputs, outputs, stage_id) + GetBackwardComputationCost(inputs, outputs, stage_id);
  }
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                    int64_t) const override;
  // Taking account for input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
  // Not taking account of output
  void CalculateOutputInMemory() override;

  void set_strategy(const Shape &strategy) { strategy_ = strategy; }
  void set_crop_size(const std::vector<int64_t> &crop_size) { crop_size_ = crop_size; }

 protected:
  Shape strategy_;
  std::vector<int64_t> crop_size_;

 private:
  static const size_t CROP_AND_RESIZE_COST_WEIGHT0 = 1;
  static const size_t CROP_AND_RESIZE_COST_WEIGHT1 = 1;
  static const size_t CROP_AND_RESIZE_COST_WEIGHT2 = 8;
  static const size_t CROP_AND_RESIZE_COST_WEIGHT3 = 2;
};

class ROIAlignCost : public CropAndResizeCost {
 public:
  ROIAlignCost() : CropAndResizeCost() {}
  ~ROIAlignCost() override = default;

  double GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                            int64_t) const override;
  double GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                   int64_t) const override;
  double GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &outputs,
                                    int64_t) const override;
  // Taking account for input
  void CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) override;
  // Taking account of output
  void CalculateOutputInMemory() override;

  void set_pooled_shape(const Shape &pooled_shape) { pooled_shape_ = pooled_shape; }

 protected:
  Shape pooled_shape_;

 private:
  static const size_t ROI_ALIGN_COST_WEIGHT0 = 1;
  static const size_t ROI_ALIGN_COST_WEIGHT1 = 4;
  static const size_t ROI_ALIGN_COST_WEIGHT2 = 2;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_OPERATOR_COSTMODEL_H_
