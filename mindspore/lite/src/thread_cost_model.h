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

#ifndef MINDSPORE_LITE_SRC_THREAD_COST_MODEL_H
#define MINDSPORE_LITE_SRC_THREAD_COST_MODEL_H

#include <stdint.h>
#include "nnacl/op_base.h"
#include "include/api/context.h"

namespace mindspore::lite {
typedef struct ThreadCostContext {
  int64_t total_unit_num_;
  int64_t per_unit_load_num_;
  int64_t per_unit_store_num_;
  float per_unit_compute_cost_;
} ThreadCostContext;

struct ThreadCostModel {
  static float unit_cost(const ThreadCostContext *thread_cost_context) {
    return per_unit_load_cost_ * thread_cost_context->per_unit_load_num_ +
           per_unit_store_cost_ * thread_cost_context->per_unit_store_num_ +
           thread_cost_context->per_unit_compute_cost_ * per_unit_compute_num_;
  }

  static float total_cost(const ThreadCostContext *thread_cost_context) {
    return thread_cost_context->total_unit_num_ * unit_cost(thread_cost_context);
  }

  // thread_num assesses parallel thread num. Value of 1.0 means ideal parallel task size. Values < 1.0 mean that task
  // granularity needs to be increased to mitigate parallelization overheads.
  static float parallel_degree(const ThreadCostContext *thread_cost_context) {
    return total_cost(thread_cost_context) / parallel_thread_cost_;
  }

  static int thread_num(const ThreadCostContext *thread_cost_context) {
    return MSMAX(
      1, static_cast<int>((total_cost(thread_cost_context) - thread_startup_cost_) / single_thread_cost_ + 0.9));
  }

  static int64_t thread_block_size(const ThreadCostContext *thread_cost_context) {
    return static_cast<int64_t>(parallel_thread_cost_ / unit_cost(thread_cost_context));
  }
  static int get_optimal_thread_num(const ThreadCostContext *thread_cost_context, const int thread_num);

  static float per_unit_load_cost_;      // per unit load cost
  static float per_unit_store_cost_;     // per unit store cost
  static int64_t per_unit_compute_num_;  // per unit compute num

  static float thread_startup_cost_;   // thread startup inherent cost
  static float single_thread_cost_;    // Minimum cost of single-threaded
  static float parallel_thread_cost_;  // Minimum cost of per thread in parallel-thread
};

int UpdateThreadNum(const Context *context, const ThreadCostContext *thread_cost_context, int task_num);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_INNER_CONTEXT_H
