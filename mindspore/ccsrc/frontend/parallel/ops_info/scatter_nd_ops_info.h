/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_ND_OPS_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_ND_OPS_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
class ScatterNdOpsInfo : public OperatorInfo {
 public:
  ScatterNdOpsInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ScatterNdOpsInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &) override;
  void ReComputeBatchSplitFlagList() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  Status InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) override;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  virtual Status CheckInputStrategy(const Dimensions &strategy_item) { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  std::vector<AnfNodePtr> PrepareReplaceGraph();
  virtual Status ComputeReplaceGraph(const CNodePtr &cnode);
  Status InferBias();
  bool do_replace_graph_ = false;
  int64_t bias_ = 0;
  int64_t slice_size_ = 0;
  size_t gather_dims_size_ = 1;
  GenerateGraph gen_g_ = GenerateGraph(attrs_);
};

using ScatterNdOpsInfoPtr = std::shared_ptr<ScatterNdOpsInfo>;

class ScatterNdAddInfo : public ScatterNdOpsInfo {
 public:
  ScatterNdAddInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterNdOpsCost>()) {}
  Status InferMirrorOps() override { return SUCCESS; }  // the scatter_nd_add only use in eval/predict
  ~ScatterNdAddInfo() override = default;
};

class ScatterNdSubInfo : public ScatterNdOpsInfo {
 public:
  ScatterNdSubInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterNdOpsCost>()) {}
  Status InferMirrorOps() override { return SUCCESS; }  // the scatter_nd_sub only use in eval/predict
  ~ScatterNdSubInfo() override = default;
};

class ScatterNdUpdateInfo : public ScatterNdOpsInfo {
 public:
  ScatterNdUpdateInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                      const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterNdOpsCost>()) {}
  Status InferMirrorOps() override { return SUCCESS; }  // the scatter_nd_update only use in eval/predict
  ~ScatterNdUpdateInfo() override = default;

 protected:
  Status CheckInputStrategy(const Dimensions &strategy_item) override;
};

class TensorScatterUpdateInfo : public ScatterNdOpsInfo {
 public:
  TensorScatterUpdateInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorScatterOpsCost>()) {}
  ~TensorScatterUpdateInfo() override = default;

 protected:
  Status CheckInputStrategy(const Dimensions &strategy_item) override;
};

class TensorScatterAddInfo : public ScatterNdOpsInfo {
 public:
  TensorScatterAddInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorScatterOpsCost>()) {}
  ~TensorScatterAddInfo() override = default;
};

class TensorScatterSubInfo : public ScatterNdOpsInfo {
 public:
  TensorScatterSubInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorScatterOpsCost>()) {}
  ~TensorScatterSubInfo() override = default;
};

class ScatterNdMulDivBaseInfo : public ScatterNdOpsInfo {
 public:
  ScatterNdMulDivBaseInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : ScatterNdOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<TensorScatterOpsCost>()) {}
  ~ScatterNdMulDivBaseInfo() override = default;

 protected:
  Status ComputeReplaceGraph(const CNodePtr &cnode) override;
};

class TensorScatterMulInfo : public ScatterNdMulDivBaseInfo {
 public:
  TensorScatterMulInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ScatterNdMulDivBaseInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~TensorScatterMulInfo() override = default;
};

class TensorScatterDivInfo : public ScatterNdMulDivBaseInfo {
 public:
  TensorScatterDivInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ScatterNdMulDivBaseInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~TensorScatterDivInfo() override = default;
};

class TensorScatterMaxInfo : public TensorScatterUpdateInfo {
 public:
  TensorScatterMaxInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : TensorScatterUpdateInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~TensorScatterMaxInfo() override = default;
};

class TensorScatterMinInfo : public TensorScatterUpdateInfo {
 public:
  TensorScatterMinInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : TensorScatterUpdateInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~TensorScatterMinInfo() override = default;
};

using ScatterAddInfoPtr = std::shared_ptr<ScatterNdAddInfo>;
using ScatterSubInfoPtr = std::shared_ptr<ScatterNdSubInfo>;
using ScatterUpdateInfoPtr = std::shared_ptr<ScatterNdUpdateInfo>;
using TensorScatterUpdateInfoPtr = std::shared_ptr<TensorScatterUpdateInfo>;
using TensorScatterAddInfoPtr = std::shared_ptr<TensorScatterAddInfo>;
using TensorScatterSubInfoPtr = std::shared_ptr<TensorScatterSubInfo>;
using TensorScatterMulInfoPtr = std::shared_ptr<TensorScatterMulInfo>;
using TensorScatterDivInfoPtr = std::shared_ptr<TensorScatterDivInfo>;
using TensorScatterMaxInfoPtr = std::shared_ptr<TensorScatterMaxInfo>;
using TensorScatterMinInfoPtr = std::shared_ptr<TensorScatterMinInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_ND_OPS_INFO_H_
