/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_QUANT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_QUANT_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/arithmetic_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class FakeQuantPerLayerInfo : public OperatorInfo {
 public:
  FakeQuantPerLayerInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                        const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<AddNCost>()) {}
  ~FakeQuantPerLayerInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferForwardCommunication() override { return SUCCESS; }
};

class FakeQuantPerChannelInfo : public FakeQuantPerLayerInfo {
 public:
  FakeQuantPerChannelInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape,
                          const PrimitiveAttrs &attrs)
      : FakeQuantPerLayerInfo(name, input_shape, output_shape, attrs) {}
  ~FakeQuantPerChannelInfo() = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  int64_t channel_axis_ = -1;
};

class MinMaxUpdatePerLayerInfo : public FakeQuantPerLayerInfo {
 public:
  MinMaxUpdatePerLayerInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                           const PrimitiveAttrs &attrs)
      : FakeQuantPerLayerInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~MinMaxUpdatePerLayerInfo() override = default;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;  // it has two outputs
  Status InferAsLossDivisor() override;
  Status InferForwardGroup();
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

  bool ema_ = False;
  float ema_decay_ = 0.0;
  std::string forward_group_;
  std::string op_name_;
  OperatorAttrs op_attrs_;
};

class MinMaxUpdatePerChannelInfo : public MinMaxUpdatePerLayerInfo {
 public:
  MinMaxUpdatePerChannelInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                             const PrimitiveAttrs &attrs)
      : MinMaxUpdatePerLayerInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~MinMaxUpdatePerChannelInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override;
  Status InferTensorMap() override;  // it has two outputs
  Status InferAsLossDivisor() override;
  Status InferForwardGroup();
  int64_t channel_axis_ = -1;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_QUANT_INFO_H_
