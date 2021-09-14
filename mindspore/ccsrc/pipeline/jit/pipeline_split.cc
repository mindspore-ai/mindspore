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

#include <string>
#include <memory>
#include "pipeline/jit/pipeline_split.h"
#include "utils/ms_context.h"
#include "utils/comm_manager.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace pipeline {
std::string GetWorldGroup() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = parallel::HCCL_WORLD_GROUP;
  } else if (backend == kGPUDevice) {
    world_group = parallel::NCCL_WORLD_GROUP;
  } else {
    MS_LOG(EXCEPTION) << "Invalid backend: " << backend;
  }
  return world_group;
}

static int64_t GetRank() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto world_group = GetWorldGroup();
  int64_t global_rank = parallel::ParallelContext::GetInstance()->global_rank();
  uint32_t rank_id = 0;
  if (!parallel::ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed.";
    }
    global_rank = UintToInt(rank_id);
  }
  return global_rank;
}

static int64_t InferStage(int64_t rank_id, int64_t stage_num, int64_t device_num) {
  if (stage_num == 0) {
    MS_LOG(EXCEPTION) << "stage_num is zero";
  }
  if (device_num % stage_num != 0) {
    MS_LOG(EXCEPTION) << "Device_num must be divisible by the stage_num, got device_num: " << device_num
                      << "stage_num: " << stage_num;
  }
  auto per_stage_rank_num = device_num / stage_num;
  return rank_id / per_stage_rank_num;
}

// Only auto_parallel and semi_auto_parallel support PipelineSplit
bool PipelineSplit(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::SEMI_AUTO_PARALLEL && parallel_mode != parallel::AUTO_PARALLEL) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support pipeline split.";
    return true;
  }
  auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (stage_num <= 1) {
    MS_LOG(INFO) << "stage num is: " << stage_num << ". No need Pipeline split.";
    return true;
  }
  auto manager = res->manager();
  auto root = res->func_graph();
  auto global_rank = GetRank();
  auto world_group = GetWorldGroup();
  uint32_t world_rank_size = 0;
  int64_t device_num = 0;
  if (!parallel::ParallelContext::GetInstance()->device_num_is_set()) {
    if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
      MS_LOG(EXCEPTION) << "Get rank size failed";
    }
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  } else {
    device_num = parallel::ParallelContext::GetInstance()->device_num();
  }
  if (device_num < 1) {
    MS_LOG(EXCEPTION) << "Invalid device num: " << device_num;
  }
  if (global_rank < 0) {
    MS_LOG(EXCEPTION) << "Invalid global rank: " << global_rank;
  }
  auto stage = InferStage(global_rank, stage_num, device_num);
  auto per_stage_rank_num = device_num / stage_num;
  if (parallel::ParallelInit() != parallel::SUCCESS) {
    MS_LOG(EXCEPTION) << "parallel init failed.";
  }
  auto transformer =
    std::make_shared<parallel::PipelineTransformer>(manager, stage, root, global_rank, per_stage_rank_num);
  // step1: Do color graph
  transformer->Coloring();
  transformer->MainGraph();
  // step2: Do color broadcast
  transformer->BroadCastColoring();
  transformer->LabelMicroBatch();
  // step3: Handle shared parameters
  transformer->ParameterColoring();
  // step4: Cut Graph
  transformer->CutGraph();
  // step5: Handle Sens
  if (root->has_flag(parallel::TRAINING)) {
    transformer->CoverSensShape();
  }
  // step6: Elim Graph stages and no used parameter
  transformer->ElimGraphStage();
  transformer->ElimParameter();
  return true;
}
}  // namespace pipeline
}  // namespace mindspore
