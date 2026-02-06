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
#include <tuple>

#include "backend/ms_backend/runtime/actors/base/abstract_actor.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"
#include "backend/ms_backend/runtime/actors/base/data_prepare_actor.h"
#include "backend/ms_backend/runtime/actors/base/data_source_actor.h"
#include "backend/ms_backend/runtime/actors/base/loop_count_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/super_kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "backend/ms_backend/runtime/actors/heterogeneous/copy_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/fusion_actor.h"
#include "backend/ms_backend/runtime/actors/base/memory_alloc_actor.h"
#include "backend/ms_backend/runtime/actors/base/memory_free_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/control_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/switch_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/gather_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/entrance_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/exit_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/stack_actor.h"
#include "backend/ms_backend/runtime/graph_scheduler/control_flow/control_node_scheduler.h"
#include "backend/ms_backend/runtime/actors/control_flow/condition_gather_runner.h"
#include "backend/ms_backend/runtime/actors/control_flow/condition_switch_runner.h"

namespace mindspore {
namespace runtime {
void DumpParameterStore(std::ofstream &ofs);
void DumpContinuousMemoryNodes(const ActorSet *actor_set, std::ofstream &ofs);
void DumpDataPrepareActor(const DataPrepareActorPtr &actor, std::ofstream &ofs);
void DumpLoopCountActor(const LoopCountActorPtr &actor, std::ofstream &ofs);
void DumpOutputActor(const OutputActorPtr &actor, std::ofstream &ofs);
void DumpDSActors(const std::vector<DataSourceActorPtr> &actors, std::ofstream &ofs);
void DumpKernelActors(const std::vector<KernelActorPtr> &actors, std::ofstream &ofs);
void DumpSuperKernelActors(const std::vector<SuperKernelActorPtr> &actors, std::ofstream &ofs);
void DumpAnyTypeKernelActors(const std::vector<AnyTypeKernelActorPtr> &actors, std::ofstream &ofs);
void DumpNoInputKernelActors(const std::vector<AbstractActorPtr> &actors, std::ofstream &ofs);
void DumpMemoryActors(const std::vector<MemoryAwareActorPtr> &actors, std::ofstream &ofs);
void DumpCopyActors(const std::vector<CopyActorPtr> &actors, std::ofstream &ofs);
void DumpFusionActors(const std::vector<FusionActorPtr> &actors, std::ofstream &ofs);
void DumpControlActors(const ControlActorSetPtr &control_actor_set, std::ofstream &ofs);
using DeviceAddressPtr = device::DeviceAddressPtr;
using KernelTensorPtr = kernel::KernelTensorPtr;
using ActorInfoMap = mindspore::HashMap<AbstractActor *, std::tuple<size_t, std::vector<KernelTensorPtr>>>;
using GetInputAidFunc = std::function<std::vector<std::string>(AbstractActor *const)>;
std::vector<AbstractActor *> TopoSortForActor(AbstractActor *root, const GetInputAidFunc &get_input_func = nullptr);
void DumpActorInfo(AbstractActor *actor, size_t index, ActorInfoMap *actor_info, std::ofstream &ofs);
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_DUMP_H_
