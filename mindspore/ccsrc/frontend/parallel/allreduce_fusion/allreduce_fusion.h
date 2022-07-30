/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_

#include <string>
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "frontend/parallel/allreduce_fusion/allreduce_graph.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
constexpr int64_t DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALGORITHM = 0;
constexpr int64_t DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TIMES = 0;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_PERCENT = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_TIME = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_INHERENT_TIME = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_BANDWIDTH = 0.1;
constexpr double DEFAULT_COST_MODEL_ALLREDUCE_FUSION_COMPUTATION_TIME_PARAMETER = 0.1;
constexpr int64_t DEFAULT_THRESHOLD_MB_TO_BYTE = 262144;

const uint64_t MAX_RECURSIVE_CALL_TIMES = 100;
class AllCommFusion {
 public:
  AllCommFusion()
      : allreduce_graph_(),
        ret_(nullptr),
        forward_ret_(nullptr),
        root_graph_(nullptr),
        tail_time_(0),
        computation_time_parameter_(0) {}
  virtual ~AllCommFusion() = default;
  Status ProcessCommOpsFusion(const CNodePtr &ret, const std::string &comm_name);

 private:
  Status SetFusionBySize(const CNodePtr &ret, int64_t threshold, const PrimitivePtr &primp) const;
  Status SetFusionBySizeReduceScatter(const CNodePtr &ret, int64_t threshold, const PrimitivePtr &primp) const;
  AllreduceGraph allreduce_graph_;
  CNodePtr ret_;
  CNodePtr forward_ret_;
  FuncGraphPtr root_graph_;
  double tail_time_;
  double computation_time_parameter_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_FUSION_H_
