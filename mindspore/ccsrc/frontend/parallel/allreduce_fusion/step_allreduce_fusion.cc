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

#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include <ctime>
#include <string>
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/allreduce_fusion/allreduce_fusion.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/status.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
bool StepAllreduceFusion(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  bool enable_all_reduce_fusion = ParallelContext::GetInstance()->enable_all_reduce_fusion();
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!root->has_flag(AUTO_PARALLEL) || ((parallel_mode != AUTO_PARALLEL) && (parallel_mode != SEMI_AUTO_PARALLEL)) ||
      (!enable_all_reduce_fusion) || (root->has_flag(ALLREDUCE_FUSION_RUN_ONCE_ONLY))) {
    return changes;
  }
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time {
    0
  }, end_time{0};
  (void)gettimeofday(&start_time, nullptr);
#endif
  MS_LOG(INFO) << "Now entering allreduce fusion";
  DumpGraph(root, std::string(ALLREDUCE_FUSION_BEGIN));

  pipeline::ResourceBasePtr res = optimizer->resource();
  MS_EXCEPTION_IF_NULL(res);

  FuncGraphManagerPtr manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  CNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);

  AllreduceFusion allreduce_fusion;
  if (allreduce_fusion.ProcessAllreduceFusion(ret) != SUCCESS) {
    MS_LOG(EXCEPTION) << "ProcessAllreduceFusion failed";
  }

  DumpGraph(root, std::string(ALLREDUCE_FUSION_END));

  // allreduce fusion only run once
  root->set_flag(ALLREDUCE_FUSION_RUN_ONCE_ONLY, true);
  res->results()[pipeline::kStepParallelGraph] = root;
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
  MS_LOG(INFO) << "Now leaving allreduce fusion, used time: " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = 1000000 * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Now leaving allreduce fusion, used time: " << time << " us";
#endif
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
