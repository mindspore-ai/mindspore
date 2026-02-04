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

#include "frontend/parallel/parallel_whole_graph_processor.h"

#include <vector>
#include <string>
#include <unordered_map>

#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/parallel_processor_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/fold_pipeline_split_utils.h"
#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/graph_util/grad_accumulation_utils.h"
#include "frontend/parallel/interleaved_parallel/interleaved_parallel.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/parallel_context.h"
#include "include/utils/comm_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
namespace {
static void ReorderForPipelineSplit(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  if (is_pp_interleave) {
    return;
  }
  auto pipeline_stages = parallel_context->pipeline_stage_split_num();
  if (!root->has_flag(kSkipAutoParallelCompile) && !root->has_flag(BACKWARD) && pipeline_stages > 1) {
    root->set_flag(BACKWARD, true);
    if (IsTraining(manager)) {
      if (parallel_context->enable_fold_pipeline()) {
        MS_LOG(INFO) << "Begin Fold Pipeline Reorder. ";
        FoldPipelineReorder(root);
      } else {
        Reorder(root);
      }
    } else {
      ReorderForPredict(root, manager);
    }
  }
}

static void ReorderForGradAccumulation(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  if (!root->has_flag(kSkipAutoParallelCompile) && !root->has_flag(BACKWARD) &&
      ParallelContext::GetInstance()->grad_accumulation_step() > 1) {
    root->set_flag(BACKWARD, true);
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
    DumpGraph(root, "before_reorder");
    if (IsTraining(manager)) {
      if (cell_reuse) {
        TagMicroBatchBpEndInCellShare(root, manager);
      }
      std::unordered_map<int64_t, std::vector<CNodePtr>> forward_start;
      std::unordered_map<int64_t, std::vector<CNodePtr>> backward_end;
      ExtractMicroBatchBorderNodes(root, &forward_start, &backward_end);
      ReorderGradAccumulation(root, forward_start, backward_end);
      DumpGraph(root, "after_reorder");
    } else {
      MS_LOG(EXCEPTION) << "Current not support predict with grad_accu";
    }
  }
}

}  // namespace

void ParallelWholeGraphProcessor::Process() {
  ReorderForPipelineSplit(processor_context_->root, processor_context_->manager);
  ReorderForGradAccumulation(processor_context_->root, processor_context_->manager);
}
}  // namespace parallel
}  // namespace mindspore
