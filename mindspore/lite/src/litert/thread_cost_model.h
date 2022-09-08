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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_THREAD_COST_MODEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_THREAD_COST_MODEL_H_

#include <stdint.h>
#include "nnacl/op_base.h"
#include "include/api/context.h"
#include "schema/ops_generated.h"

namespace mindspore::lite {
typedef struct ThreadCostContext {
  int64_t total_unit_num_;
  int64_t per_unit_load_num_;
  int64_t per_unit_store_num_;
  float per_unit_compute_cost_;
} ThreadCostContext;

#define TC_PTYPE(primitive_type) (primitive_type << 16)
#define TC_ATYPE(activation_type) (activation_type)
#define TC_TYPE(primitive_type, activation_type) (TC_PTYPE(primitive_type) + TC_ATYPE(activation_type))

struct ThreadCostModel {
  static float UnitCost(const ThreadCostContext *thread_cost_context) {
    return per_unit_load_cost_ * thread_cost_context->per_unit_load_num_ +
           per_unit_store_cost_ * thread_cost_context->per_unit_store_num_ +
           thread_cost_context->per_unit_compute_cost_ * per_unit_compute_num_;
  }

  static float TotalCost(const ThreadCostContext *thread_cost_context) {
    return thread_cost_context->total_unit_num_ * UnitCost(thread_cost_context);
  }

  // ThreadNum assesses parallel thread num. Value of 1.0 means ideal parallel task size. Values < 1.0 mean that task
  // granularity needs to be increased to mitigate parallelization overheads.
  static float ParallelDegree(const ThreadCostContext *thread_cost_context) {
    return TotalCost(thread_cost_context) / parallel_thread_cost_;
  }

  static int ThreadNum(const ThreadCostContext *thread_cost_context) {
    return MSMAX(1,
                 static_cast<int>((TotalCost(thread_cost_context) - thread_startup_cost_) / single_thread_cost_ + 0.9));
  }

  static int64_t ThreadBlockSize(const ThreadCostContext *thread_cost_context) {
    return static_cast<int64_t>(parallel_thread_cost_ / UnitCost(thread_cost_context));
  }
  static int GetOptimalThreadNum(const ThreadCostContext *thread_cost_context, const int thread_num);

  static float per_unit_load_cost_;      // per unit load cost
  static float per_unit_store_cost_;     // per unit store cost
  static int64_t per_unit_compute_num_;  // per unit compute num

  static float thread_startup_cost_;   // thread startup inherent cost
  static float single_thread_cost_;    // Minimum cost of single-threaded
  static float parallel_thread_cost_;  // Minimum cost of per thread in parallel-thread
};

float GetKernelComputeCost(int32_t kernel_type);
int ThreadNumUpdateStrategy(const ThreadCostContext *thread_cost_context, int task_num);

#ifdef DYNAMIC_THREAD_DISTRIBUTE
int UpdateThreadNum(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num, int64_t unit_num,
                    int thread_num);
#else
inline int UpdateThreadNum(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num, int64_t unit_num,
                           int thread_num) {
  (void)kernel_type;
  (void)per_unit_load_num;
  (void)per_unit_store_num;
  (void)unit_num;
  return thread_num;
}
#endif
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_INNER_CONTEXT_H
