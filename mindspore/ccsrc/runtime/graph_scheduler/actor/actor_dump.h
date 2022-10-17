/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_DUMP_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_DUMP_H_

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "runtime/graph_scheduler/actor/abstract_actor.h"
#include "runtime/graph_scheduler/actor/data_prepare_actor.h"
#include "runtime/graph_scheduler/actor/data_source_actor.h"
#include "runtime/graph_scheduler/actor/loop_count_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/copy_actor.h"
#include "runtime/graph_scheduler/actor/custom_actor.h"
#include "runtime/graph_scheduler/actor/fusion/fusion_actor.h"
#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"
#include "runtime/graph_scheduler/actor/memory/memory_alloc_actor.h"
#include "runtime/graph_scheduler/actor/memory/memory_free_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/control_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/switch_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/gather_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/exit_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/stack_actor.h"
#include "runtime/graph_scheduler/control_node_scheduler.h"

namespace mindspore {
namespace runtime {
void DumpDataPrepareActor(const DataPrepareActorPtr &actor, std::ofstream &ofs);
void DumpLoopCountActor(const LoopCountActorPtr &actor, std::ofstream &ofs);
void DumpOutputActor(const OutputActorPtr &actor, std::ofstream &ofs);
void DumpDSActors(const std::vector<DataSourceActorPtr> &actors, std::ofstream &ofs);
void DumpKernelActors(const std::vector<KernelActorPtr> &actors, std::ofstream &ofs);
void DumpSuperKernelActors(const std::vector<SuperKernelActorPtr> &actors, std::ofstream &ofs);
void DumpNoInputKernelActors(const std::vector<AbstractActorPtr> &actors, std::ofstream &ofs);
void DumpMemoryActors(const std::vector<MemoryAwareActorPtr> &actors, std::ofstream &ofs);
void DumpCopyActors(const std::vector<CopyActorPtr> &actors, std::ofstream &ofs);
void DumpFusionActors(const std::vector<FusionActorPtr> &actors, std::ofstream &ofs);
void DumpControlActors(const ControlActorSetPtr &control_actor_set, std::ofstream &ofs);
void DumpCustomActors(const std::vector<CustomActorPtr> &actors, std::ofstream &ofs);
void DumpSwapActors(const std::vector<std::vector<MemSwapActorPtr>> &actors, std::ofstream &ofs);
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_DUMP_H_
