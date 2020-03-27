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

#include "optimizer/parallel/auto_parallel/costmodel.h"
#include <utility>
#include <numeric>
#include <cmath>
#include "optimizer/parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
void Simplify(CostPtrList* clist_ptrs) {
  // Sort the cost_list with the memory_cost increasing, and communication_cost decreasing order. This method
  // excludes the cost with greater memory_cost and greater communication_cost.
  // E.g. clist_ptrs = {<100, 20>, <200, 10>, <300, 50>}. After this method, clist_ptrs = {<200, 10>, <100, 20>}
  if (!COST_MODEL_SIMPLIFY_CALCULATION) {
    return;
  }
  MS_EXCEPTION_IF_NULL(clist_ptrs);
  std::vector<size_t> id(clist_ptrs->size());
  std::iota(id.begin(), id.end(), size_t(0));
  std::sort(id.begin(), id.end(), [&clist_ptrs](size_t x, size_t y) {
    return clist_ptrs->at(x)->memory_cost_ < clist_ptrs->at(y)->memory_cost_;
  });
  CostPtrList ret;
  for (size_t i = 0; i < clist_ptrs->size(); ++i) {
    if ((ret.size() == size_t(0)) || (clist_ptrs->at(id[i])->communication_cost_ < ret.back()->communication_cost_)) {
      ret.emplace_back(std::move(clist_ptrs->at(id[i])));
    }
  }
  *clist_ptrs = std::move(ret);
}

void SimplifyForDreasingCommunicationWithPartialPara(CostPtrList* clist_ptrs) {
  // Sort the cost_list with the memory_cost increasing, and communication_with_partial_para_cost decreasing order.
  // This method excludes the cost with greater memory_cost and greater communication_without_para_cost.
  if (!COST_MODEL_SIMPLIFY_CALCULATION) {
    return;
  }
  MS_EXCEPTION_IF_NULL(clist_ptrs);
  std::vector<size_t> id(clist_ptrs->size());
  std::iota(id.begin(), id.end(), size_t(0));
  std::sort(id.begin(), id.end(), [&clist_ptrs](size_t x, size_t y) {
    return clist_ptrs->at(x)->memory_cost_ < clist_ptrs->at(y)->memory_cost_;
  });
  CostPtrList ret;
  for (size_t i = 0; i < clist_ptrs->size(); ++i) {
    if ((ret.size() == size_t(0)) ||
        (clist_ptrs->at(id[i])->communication_with_partial_para_ < ret.back()->communication_with_partial_para_)) {
      ret.emplace_back(std::move(clist_ptrs->at(id[i])));
    }
  }
  *clist_ptrs = std::move(ret);
}

void RefineForPracticalCost(const CostPtr& origin_cost, bool is_redistribution) {
  MS_EXCEPTION_IF_NULL(origin_cost);
  if (is_redistribution) {
    // Redistribution cost
    if ((origin_cost->communication_redis_forward_ > EPS) &&
        (origin_cost->communication_redis_forward_ <= COST_MODEL_COMMUNI_THRESHOLD)) {
      origin_cost->communication_redis_forward_ = COST_MODEL_COMMUNI_CONST;
    } else if (origin_cost->communication_redis_forward_ > COST_MODEL_COMMUNI_THRESHOLD) {
      origin_cost->communication_redis_forward_ += COST_MODEL_COMMUNI_BIAS;
    }
    if ((origin_cost->communication_redis_backward_ > EPS) &&
        (origin_cost->communication_redis_backward_ <= COST_MODEL_COMMUNI_THRESHOLD)) {
      origin_cost->communication_redis_backward_ = COST_MODEL_COMMUNI_CONST;
    } else if (origin_cost->communication_redis_backward_ > COST_MODEL_COMMUNI_THRESHOLD) {
      origin_cost->communication_redis_backward_ += COST_MODEL_COMMUNI_BIAS;
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
        (origin_cost->communication_without_parameter_ <= COST_MODEL_COMMUNI_THRESHOLD)) {
      origin_cost->communication_without_parameter_ = COST_MODEL_COMMUNI_CONST;
    } else if (origin_cost->communication_without_parameter_ > COST_MODEL_COMMUNI_THRESHOLD) {
      origin_cost->communication_without_parameter_ += COST_MODEL_COMMUNI_BIAS;
    }
    // total
    if (origin_cost->communication_cost_ > EPS) {
      origin_cost->communication_cost_ = origin_cost->communication_without_parameter_ + backward;
    }
    if (origin_cost->communication_with_partial_para_ > EPS) {
      origin_cost->communication_with_partial_para_ =
        origin_cost->communication_without_parameter_ + COST_MODEL_GAMMA * backward;
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
