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

#ifndef MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_
#define MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_

#include <unordered_map>
#include <vector>
#include "ir/anf.h"
#include "parallel/allreduce_fusion/allreduce_graph.h"
#include "parallel/status.h"

namespace mindspore {
namespace parallel {
using CNodeCostMap = std::unordered_map<CNodePtr, double>;

constexpr int32_t DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALGORITHM = 0;
constexpr int32_t DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TIMES = 0;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_PERCENT = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_TIME = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_INHERENT_TIME = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_BANDWIDTH = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_COMPUTATION_TIME_PARAMETER = 0.1;

constexpr char FUSION[] = "fusion";
constexpr char PARAMETER[] = "parameter";
const uint32_t MAX_RECURSIVE_CALL_TIMES = 100;
const double FUSION_COST_EPS = 1e-7;
class AllreduceFusion {
 public:
  AllreduceFusion()
      : allreduce_graph_(),
        ret_(nullptr),
        forward_ret_(nullptr),
        root_graph_(nullptr),
        tail_time_(0),
        allreduce_inherent_time_(0),
        allreduce_bandwidth_(0),
        computation_time_parameter_(0) {}
  virtual ~AllreduceFusion() = default;
  Status ProcessAllreduceFusion(const CNodePtr& ret);

 private:
  Status AddNodeToGraph();
  CNodeCostMap FindCNode(const AnfNodePtr& from, uint32_t recursive_times = 0) const;
  CNodeCostMap FindNextCNodes(const CNodePtr& from, uint32_t recursive_times = 0) const;
  Status AddEdgeToGraph();
  std::vector<double> GenerateCostMap(int32_t fusion_times, double tail_percent) const;
  Status SetFusion(const std::vector<double>& cost_map);
  Status SetFusionByAlgorithm(int32_t algorithm);
  Status SetFusionByBackwardCompTime();
  Status SetFusionByBackwardCompAndAllreduceTime();
  Status GetSetFusionByBackwardCompAndAllreduceTimeParams();

  AllreduceGraph allreduce_graph_;
  CNodePtr ret_;
  CNodePtr forward_ret_;
  FuncGraphPtr root_graph_;
  double tail_time_;
  double allreduce_inherent_time_;
  double allreduce_bandwidth_;
  double computation_time_parameter_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_
