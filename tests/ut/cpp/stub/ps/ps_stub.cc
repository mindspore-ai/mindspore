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

#include "ps/ps_cache/ps_cache_manager.h"
#include "ps/util.h"
#include "ps/worker.h"
#include "ps/scheduler.h"
#include "ps/parameter_server.h"

namespace mindspore {
namespace ps {
PsCacheManager &PsCacheManager::GetInstance() {
  static PsCacheManager instance{};
  return instance;
}

void PsCacheManager::Finalize() {}
int PsCacheManager::cache_indices_lower_bound() const { return 1; }

bool Util::IsRoleOfPServer() { return true; }
bool Util::IsRoleOfScheduler() { return true; }
bool Util::FuseServerCommOps(const pipeline::ResourcePtr &res) { return true; }

Worker &Worker::GetInstance() {
  static Worker instance{};
  return instance;
}

void Worker::Run() {}
void Worker::Finalize() {}

ParameterServer &ParameterServer::GetInstance() {
  static ParameterServer instance{};
  return instance;
}

void ParameterServer::Run(const FuncGraphPtr &func_graph) {}

Scheduler &Scheduler::GetInstance() {
  static Scheduler instance{};
  return instance;
}

void Scheduler::Run() {}
}  // namespace ps
}  // namespace mindspore