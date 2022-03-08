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

#include "src/thread_cost_model.h"
#include "src/common/log_util.h"
#include "src/inner_context.h"
#include "thread/threadpool.h"

namespace mindspore::lite {
float ThreadCostModel::per_unit_load_cost_ = 1.0 / 64 * 11;   // 64: L2 cache size, 11 : L2 cache latency on Haswell
float ThreadCostModel::per_unit_store_cost_ = 1.0 / 64 * 11;  // 64: L2 cache size, 11 : L2 cache latency on Haswell
int64_t ThreadCostModel::per_unit_compute_num_ = 1;           // 1 : per unit compute num

float ThreadCostModel::thread_startup_cost_ = 100000.0f;  // 100000 : thread startup inherent cost
float ThreadCostModel::single_thread_cost_ = 100000.0f;   // 100000 : Minimum cost of single-threaded
float ThreadCostModel::parallel_thread_cost_ = 40000.0f;  // 40000 : Minimum cost of per thread in parallel-thread

int ThreadCostModel::get_optimal_thread_num(const ThreadCostContext *thread_cost_context, const int thread_num) {
  const int64_t max_oversharding_factor = 4;

  int64_t block_size = MSVALID(max_oversharding_factor * thread_num, thread_block_size(thread_cost_context),
                               thread_cost_context->total_unit_num_);
  int64_t block_count = UP_DIV(thread_cost_context->total_unit_num_, block_size);

  int64_t max_block_size = MSMIN(thread_cost_context->total_unit_num_, 2 * block_size);
  double max_efficiency = static_cast<double>(block_count) / (UP_DIV(block_count, thread_num) * thread_num);
  for (int64_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
    int64_t cur_block_size = UP_DIV(thread_cost_context->total_unit_num_, prev_block_count - 1);
    if (cur_block_size > max_block_size) {
      break;
    }
    const int64_t cur_block_count = UP_DIV(thread_cost_context->total_unit_num_, cur_block_size);
    MS_ASSERT(cur_block_count < prev_block_count);
    prev_block_count = cur_block_count;
    const double cur_efficiency =
      static_cast<double>(cur_block_count) / (UP_DIV(cur_block_count, thread_num) * thread_num);
    if (cur_efficiency + 0.01 >= max_efficiency) {  // update threshold : 0.01
      block_size = cur_block_size;
      block_count = cur_block_count;
      if (max_efficiency < cur_efficiency) {
        max_efficiency = cur_efficiency;
      }
    }
  }
  return block_count;
}

int UpdateThreadNum(const Context *context, const ThreadCostContext *thread_cost_context, int task_num) {
  if (task_num <= 1) {
    return task_num;
  }
  ThreadPool *pool = static_cast<const lite::InnerContext *>(context)->thread_pool();
  if (pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return RET_NULL_PTR;
  }

  if (thread_cost_context != nullptr) {
    if (ThreadCostModel::thread_num(thread_cost_context) == 1) {
      return 1;
    }
    int opt_thread = static_cast<int>(ThreadCostModel::parallel_degree(thread_cost_context));
    task_num = MSVALID(1, opt_thread, task_num);
    task_num = MSMIN(task_num, thread_cost_context->total_unit_num_);
  }
  return task_num;
}
}  // namespace mindspore::lite
