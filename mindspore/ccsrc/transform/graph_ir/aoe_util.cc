/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#ifdef ENABLE_AOE
#include "aoe/external/aoe.h"
#endif
#include "transform/graph_ir/aoe_util.h"

namespace mindspore {
namespace transform {
namespace AoeOptions {
const ::ge::AscendString JOB_TYPE = ::ge::AscendString("job_type");
const ::ge::AscendString FRAMEWORK = ::ge::AscendString("framework");
const ::ge::AscendString LOG_LEVEL = ::ge::AscendString("log");
const ::ge::AscendString PRECISION_MODE = ::ge::AscendString("precision_mode");
}  // namespace AoeOptions

AoeUtil::AoeUtil() : initialize_(false) {}

AoeUtil::~AoeUtil() { MS_LOG(INFO) << "release aoeutil success."; }

void AoeUtil::Initialize() {
  if (initialize_) {
    MS_LOG(INFO) << "Aoe already initialized.";
    return;
  }
#ifdef ENABLE_AOE
  std::map<::ge::AscendString, ::ge::AscendString> globalOptions = {{AoeOptions::JOB_TYPE, ::ge::AscendString("1")}};
  const Aoe::AoeStatus status = Aoe::AoeInitialize(globalOptions);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeInitialize failed.";
  }
  MS_LOG(INFO) << "AoeInitialize success.";
#endif
  initialize_ = true;
}

void AoeUtil::Destroy() {
#ifdef ENABLE_AOE
  try {
    const Aoe::AoeStatus status = Aoe::AoeFinalize();
    if (status != Aoe::AOE_SUCCESS) {
      MS_LOG(ERROR) << "AoeFinalize failed. status is " << status;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Error occurred when exec aoe finalize. Error:" << e.what();
  } catch (...) {
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(ERROR) << "Error occurred when  exec aoe finalize. Exception name: " << exName;
  }
#endif
  MS_LOG(INFO) << "AoeFinalization success.";
  initialize_ = false;
}

AoeUtil &AoeUtil::GetInstance() {
  static AoeUtil instance{};
  return instance;
}

#ifdef ENABLE_AOE
Status AoeUtil::AoeGeGraph(::ge::Session *ge_session, const transform::DfGraphPtr &graph,
                           const std::map<::ge::AscendString, ::ge::AscendString> &tuningOptions) {
  uint64_t sessionId = 0;
  Aoe::AoeStatus status = Aoe::AoeCreateSession(sessionId);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeCreateSession failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AoeCreateSession success.";

  status = Aoe::AoeSetGeSession(sessionId, ge_session);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeSetGeSession failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeSetGeSession success.";

  status = Aoe::AoeSetTuningGraph(sessionId, *graph);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeSetGraph failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeSetGraph success.";

  status = Aoe::AoeTuningGraph(sessionId, tuningOptions);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeTuningGraph failed.";
    (void)Aoe::AoeDestroySession(sessionId);
    return FAILED;
  }
  MS_LOG(DEBUG) << "->AoeTuningGraph success.";

  status = Aoe::AoeDestroySession(sessionId);
  if (status != Aoe::AOE_SUCCESS) {
    MS_LOG(ERROR) << "AoeDestroySession failed.";
    return FAILED;
  }
  return SUCCESS;
}
#else
Status AoeUtil::AoeGeGraph(::ge::Session *, const transform::DfGraphPtr &,
                           const std::map<::ge::AscendString, ::ge::AscendString> &) const {
  return SUCCESS;
}
#endif

Status AoeUtil::AoeOnlineGeGraph(const std::shared_ptr<::ge::Session> &ge_session, const transform::DfGraphPtr &graph) {
  MS_LOG(DEBUG) << "AoeOnlineGeGraph start.";
  if (ge_session == nullptr) {
    MS_LOG(ERROR) << "sess is null";
    return FAILED;
  }

  std::map<::ge::AscendString, ::ge::AscendString> tuneOptions = {
    {AoeOptions::FRAMEWORK, ::ge::AscendString("1")},
#ifdef ASCEND_910
    {AoeOptions::PRECISION_MODE, ::ge::AscendString("allow_fp32_to_fp16")},
#else
    {AoeOptions::PRECISION_MODE, ::ge::AscendString("must_keep_origin_dtype")},
#endif
    {AoeOptions::LOG_LEVEL, ::ge::AscendString("error")},
  };

  if (AoeGeGraph(ge_session.get(), graph, tuneOptions) != SUCCESS) {
    MS_LOG(ERROR) << "Failed to call Aoe online tuning.";
    return FAILED;
  }

  MS_LOG(DEBUG) << "AoeTuningGraph success.";
  return SUCCESS;
}

void AoeUtil::SaveOptimizedGraph(const int32_t &graph_id) { optimized_graphs_id_.insert(graph_id); }

bool AoeUtil::IsSaveOptimizedGraph(const int32_t &graph_id) const {
  auto iter_find = optimized_graphs_id_.find(graph_id);
  if (iter_find != optimized_graphs_id_.end()) {
    return true;
  }
  return false;
}

void AoeUtil::RemoveWaitOptimizedGraph(const std::set<std::string> &optimized_graph_names) {
  for (auto &graph_name : optimized_graph_names) {
    if (auto remove_iter = wait_optimize_graphs_.find(graph_name); remove_iter != wait_optimize_graphs_.end())
      (void)wait_optimize_graphs_.erase(remove_iter);
  }
  if (!wait_optimize_graphs_.empty()) {
    MS_LOG(WARNING) << "optimize_graphs_ is not empty";
  }
}

void AoeUtil::AddOptimizeGraph(const std::string &graph_name) { wait_optimize_graphs_.insert(graph_name); }

std::set<std::string> AoeUtil::GetWaitOptimizeGraph() const { return wait_optimize_graphs_; }
}  // namespace transform
}  // namespace mindspore
