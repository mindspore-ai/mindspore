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

#include "frontend/parallel/auto_parallel/costmodel.h"
#include <cmath>
#include <numeric>
#include <utility>
#include "frontend/parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
void Simplify(CostPtrList *clist_ptrs) {
  const auto run_phase = CostModelContext::GetInstance()->run_phase();
  if (run_phase == TRAINING_PHASE) {
    // training phase
    SimplifyForDecreasingCommunicationWithPartialPara(clist_ptrs);
  } else {
    // inference phase
    SimplifyForDecreasingCommunicationForward(clist_ptrs);
  }
}
void SimplifyForDecreasingCommunicationForward(CostPtrList *clist_ptrs) {
  // Sort the cost_list with the computation_cost_ increasing, and communication_forward decreasing order. This method
  // excludes the cost with greater computation_cost_ and greater communication_forward.
  // E.g. clist_ptrs = {<100, 20>, <200, 10>, <300, 50>}. After this method, clist_ptrs = {<200, 10>, <100, 20>}
  const auto simplify_cal = CostModelContext::GetInstance()->costmodel_simplify_cal();
  if (!simplify_cal) {
    return;
  }
  MS_EXCEPTION_IF_NULL(clist_ptrs);
  std::vector<size_t> id(clist_ptrs->size());
  std::iota(id.begin(), id.end(), size_t(0));
  std::sort(id.begin(), id.end(), [&clist_ptrs](size_t x, size_t y) {
    return clist_ptrs->at(x)->computation_cost_ < clist_ptrs->at(y)->computation_cost_;
  });
  CostPtrList ret;
  for (size_t i = 0; i < clist_ptrs->size(); ++i) {
    if ((ret.size() == size_t(0)) ||
        (clist_ptrs->at(id[i])->communication_forward_ < ret.back()->communication_forward_)) {
      (void)ret.emplace_back(std::move(clist_ptrs->at(id[i])));
    }
  }
  *clist_ptrs = std::move(ret);
}

void SimplifyForDecreasingCommunicationWithPartialPara(CostPtrList *clist_ptrs) {
  // Sort the cost_list with the computation_cost_ increasing, and communication_with_partial_para_cost decreasing
  // order. This method excludes the cost with greater computation_cost_ and greater communication_without_para_cost.
  const auto simplify_cal = CostModelContext::GetInstance()->costmodel_simplify_cal();
  if (!simplify_cal) {
    return;
  }
  MS_EXCEPTION_IF_NULL(clist_ptrs);
  std::vector<size_t> id(clist_ptrs->size());
  std::iota(id.begin(), id.end(), size_t(0));
  std::sort(id.begin(), id.end(), [&clist_ptrs](size_t x, size_t y) {
    return clist_ptrs->at(x)->computation_cost_ < clist_ptrs->at(y)->computation_cost_;
  });
  CostPtrList ret;
  for (size_t i = 0; i < clist_ptrs->size(); ++i) {
    if ((ret.size() == size_t(0)) ||
        (clist_ptrs->at(id[i])->communication_with_partial_para_ < ret.back()->communication_with_partial_para_)) {
      (void)ret.emplace_back(std::move(clist_ptrs->at(id[i])));
    }
  }
  *clist_ptrs = std::move(ret);
}

void RefineForPracticalCost(const CostPtr &origin_cost, bool is_redistribution) {
  MS_EXCEPTION_IF_NULL(origin_cost);
  const auto comm_threshold = CostModelContext::GetInstance()->costmodel_communi_threshold();
  const auto comm_const = CostModelContext::GetInstance()->costmodel_communi_const();
  const auto comm_bias = CostModelContext::GetInstance()->costmodel_communi_bias();
  const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
  if (is_redistribution) {
    // Redistribution cost
    if ((origin_cost->communication_redis_forward_ > EPS) &&
        (origin_cost->communication_redis_forward_ <= comm_threshold)) {
      origin_cost->communication_redis_forward_ = comm_const;
    } else if (origin_cost->communication_redis_forward_ > comm_threshold) {
      origin_cost->communication_redis_forward_ += comm_bias;
    }
    if ((origin_cost->communication_redis_backward_ > EPS) &&
        (origin_cost->communication_redis_backward_ <= comm_threshold)) {
      origin_cost->communication_redis_backward_ = comm_const;
    } else if (origin_cost->communication_redis_backward_ > comm_threshold) {
      origin_cost->communication_redis_backward_ += comm_bias;
    }
    origin_cost->communication_cost_ =
      origin_cost->communication_redis_forward_ + origin_cost->communication_redis_backward_;
    origin_cost->communication_without_parameter_ = origin_cost->communication_cost_;
    origin_cost->communication_with_partial_para_ = origin_cost->communication_cost_;
  } else {
    // Operator cost
    double backward = 0.0;
    if (std::abs(origin_cost->communication_cost_ - origin_cost->communication_without_parameter_) > EPS) {
      backward = origin_cost->communication_cost_ - origin_cost->communication_without_parameter_;
    }
    // forward cost
    if ((origin_cost->communication_without_parameter_ > EPS) &&
        (origin_cost->communication_without_parameter_ <= comm_threshold)) {
      origin_cost->communication_without_parameter_ = comm_const;
    } else if (origin_cost->communication_without_parameter_ > comm_threshold) {
      origin_cost->communication_without_parameter_ += comm_bias;
    }
    // total
    if (origin_cost->communication_cost_ > EPS) {
      origin_cost->communication_cost_ = origin_cost->communication_without_parameter_ + backward;
    }
    if (origin_cost->communication_with_partial_para_ > EPS) {
      origin_cost->communication_with_partial_para_ = origin_cost->communication_without_parameter_ + gamma * backward;
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
