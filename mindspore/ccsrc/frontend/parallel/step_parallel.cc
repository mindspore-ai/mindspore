/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/step_parallel.h"

#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "utils/ms_context.h"
#include "ir/graph_utils.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/parallel_processor_context.h"
#include "frontend/parallel/parallel_whole_graph_processor.h"
#include "frontend/parallel/parallel_preprocessor.h"
#include "frontend/parallel/parallel_postprocessor.h"
#include "frontend/parallel/parallel_processor.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/jit/ps/parse/parse_base.h"

namespace mindspore {
namespace parallel {
std::set<FuncGraphPtr> ForwardGraph(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = TopoSort(ret, SuccDeeperSimple);
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);
  return graph_set;
}

bool CreateGroupsByCkptFile(const std::string &file) {
  GroupInfoMap group_info_map;
  if (StrategyCheckpoint::GetInstance().LoadGroupInfo(file, &group_info_map) != SUCCESS) {
    return false;
  }

  if (CreateGroups(group_info_map) != SUCCESS) {
    return false;
  }
  MS_LOG(INFO) << "Create groups by checkpoint file success";
  return true;
}

static void SetDataParallelGroupInfo() {
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == kDataParallel) {
    auto group_info_save_path = common::GetEnv("GROUP_INFO_FILE");
    if (!group_info_save_path.empty()) {
      std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info;
      int64_t device_num = GetCommInfo().device_num;
      RankList comm_group;
      for (size_t i = 0; i < size_t(device_num); ++i) {
        comm_group.push_back(i);
      }
      ParallelContext::GetInstance()->set_group_ckpt_save_file(group_info_save_path);
      if (StrategyCheckpoint::GetInstance().SaveGroupInfo(group_info, comm_group) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Save group info failed";
      }
    }
  }
}

static void ParallelInit(const ParallelProcessorContextPtr &processor_context) {
  if (processor_context->pipeline_stages > 1) {
    return;
  }

  if (processor_context->parallel_mode == kAutoParallel) {
    return;
  }

  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }
}

bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  auto processor_context = std::make_shared<ParallelProcessorContext>(root);
  processor_context->Init(optimizer);

  SetDataParallelGroupInfo();
  if (IsTraining(processor_context->manager)) {
    root->set_flag(kTraining, true);
  }

  if (root->has_attr(mindspore::parse::CELL_COMPILE_PHASE)) {
    const auto &compile_phase = GetValue<std::string>(root->get_attr(mindspore::parse::CELL_COMPILE_PHASE));
    MS_LOG(DEBUG) << "Current compile_phase is " << compile_phase;
    StrategyLayout::GetInstance()->SetCellPhase(compile_phase);
  }

  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root)) {
    if (!root->has_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY)) {
      MS_LOG(INFO) << "Strategies would be ignored in " << processor_context->parallel_mode
                   << ", shard() only valid in [semi_]auto_parallel.";
      root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);
      MS_EXCEPTION_IF_NULL(processor_context);
      const auto &all_nodes = processor_context->all_nodes;
      CheckpointOnline(all_nodes, root);
    }
    return false;
  }

  const auto &root_all_cnodes = root->GetOrderedCnodes();
  if (root->has_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY)) {
    if (std::find_if(root_all_cnodes.begin(), root_all_cnodes.end(), [](const CNodePtr &cnode) {
          return IsPrimitiveCNode(cnode, prim::kPrimJ);
        }) == root_all_cnodes.end()) {
      // whole graph (forward and backward) process
      auto whole_graph_processor = std::make_shared<ParallelWholeGraphProcessor>(processor_context);
      whole_graph_processor->Process();
    }
    return false;
  }

  MSLogTime msTime;
  msTime.Start();
  DumpGraph(root, std::string(STEP_PARALLEL_BEGIN));

  ParallelInit(processor_context);

  auto preprocessor = std::make_shared<ParallelPreprocessor>(processor_context);
  auto processor = std::make_shared<ParallelProcessor>(processor_context);
  auto postprocessor = std::make_shared<ParallelPostprocessor>(processor_context);

  // init parallel info
  preprocessor->Process();

  // ForwardCommunication BackwardCommunication TensorRedistribution
  processor->Process();

  // forward process with inserted parallel communication op
  postprocessor->Process();

  DumpGraph(root, std::string(STEP_PARALLEL_END));
  // Keep all func graph for parallel before save result.
  SetReserved(root);

  auto res = processor_context->resource;
  MS_EXCEPTION_IF_NULL(res);
  res->SetResult(pipeline::kStepParallelGraph, root);

  // step parallel only run once
  root->set_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  // in auto parallel mode, no need to check if strategies set
  root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);

  msTime.End();
  uint64_t time = msTime.GetRunTimeUS();
  MS_LOG(INFO) << "Now leaving step parallel, used time: " << time << " us";
  return false;
}
}  // namespace parallel
}  // namespace mindspore
