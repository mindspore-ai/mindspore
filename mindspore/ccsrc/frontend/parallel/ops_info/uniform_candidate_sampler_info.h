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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNFORM_CANDIDATE_SAMPLER_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNFORM_CANDIDATE_SAMPLER_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t UNIFORM_CANDIDATE_SAMPLER_INPUTS_SIZE = 2;
class UniformCandidateSamplerInfo : public OperatorInfo {
 public:
  UniformCandidateSamplerInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                              const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs,
                     std::make_shared<UniformCandidateSamplerCost>()),
        num_sampled_(0),
        num_true_(0),
        unique_(false),
        range_max_(0),
        seed_(0),
        remove_accidental_hits_(false) {}
  ~UniformCandidateSamplerInfo() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t) override;
  std::shared_ptr<Strategys> GenerateBatchStrategies() override;
  Status SetCostUnderStrategy(const StrategyPtr &) override;
  Status InferAsLossDivisor() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status ComputeReplaceGraph(const CNodePtr &cnode);

 private:
  Status GetUniformSamplerAttrBool(const std::string &argsy, bool *value);
  Status GetUniformSamplerAttrInt64(const std::string &args, int64_t *value);
  int64_t num_sampled_;
  int64_t num_true_;
  bool unique_;
  int64_t range_max_;
  int64_t seed_;
  bool remove_accidental_hits_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNFORM_CANDIDATE_SAMPLER_INFO_H_
