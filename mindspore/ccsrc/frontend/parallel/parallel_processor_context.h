/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_CONTEXT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_CONTEXT_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "include/utils/parallel_context.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/parallel_processor_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/interleaved_parallel/interleaved_parallel.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
struct ParallelProcessorContext {
  explicit ParallelProcessorContext(const FuncGraphPtr &input_root) : root(input_root) {}

  void Init(const opt::OptimizerPtr &optimizer) {
    if (optimizer == nullptr) {
      manager = root->manager();
      resource = std::make_shared<pipeline::Resource>();
      resource->set_manager(manager);
    } else {
      resource = optimizer->resource();
      MS_EXCEPTION_IF_NULL(resource);
      manager = resource->manager();
    }

    MS_EXCEPTION_IF_NULL(manager);
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);

    is_pp_interleave = parallel_context->pipeline_interleave();
    parallel_mode = parallel_context->parallel_mode();
    pipeline_stages = parallel_context->pipeline_stage_split_num();
  }

  const FuncGraphPtr &root;
  FuncGraphManagerPtr manager{nullptr};
  pipeline::ResourceBasePtr resource{nullptr};
  bool is_pp_interleave{false};
  bool is_apply_adasum{false};
  std::string parallel_mode;
  int64_t pipeline_stages{1};
  std::vector<AnfNodePtr> all_nodes;
  std::shared_ptr<PipelinePostProcess> pipeline_processor{nullptr};
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> adasum_param_tensor_layout_map;
};

using ParallelProcessorContextPtr = std::shared_ptr<ParallelProcessorContext>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_CONTEXT_H_
