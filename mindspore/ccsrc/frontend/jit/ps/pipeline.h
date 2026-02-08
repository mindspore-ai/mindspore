/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PIPELINE_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PIPELINE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <list>
#include <optional>

#include "pybind11/pybind11.h"

#include "base/base.h"
#include "frontend/jit/ps/action.h"
#include "utils/ms_exception.h"

namespace mindspore {
// namespace to support pipeline structures definition
namespace distributed {
namespace cluster {
class TCPStoreClient;
using TCPStoreClientPtr = std::shared_ptr<TCPStoreClient>;
}  // namespace cluster
}  // namespace distributed
namespace pipeline {

namespace py = pybind11;

constexpr auto kActualArgumentIndex = "argument_index";

class Pipeline {
 public:
  Pipeline(const ResourcePtr &res, const std::vector<ActionItem> &actions) : resource_(res), actions_(actions) {}

  ~Pipeline() = default;

  void Run();

  ResourcePtr resource() { return resource_; }

 private:
  ResourcePtr resource_;
  std::vector<ActionItem> actions_;
};

class JitCompilingScope {
 public:
  JitCompilingScope() { MsContext::GetInstance()->set_jit_status(kJitCompiling); }
  ~JitCompilingScope() { MsContext::GetInstance()->set_jit_status(kNotJit); }
};

class GraphCompilingScope {
 public:
  GraphCompilingScope() {
    MsContext::GetInstance()->set_jit_status(kGraphCompiling);
    MsContext::GetInstance()->set_graph_pipeline_compiled(true);
  }
  ~GraphCompilingScope() { MsContext::GetInstance()->set_jit_status(kNotJit); }
};

class JitRunningScope {
 public:
  JitRunningScope() { MsContext::GetInstance()->set_jit_status(kJitRunning); }
  ~JitRunningScope() { MsContext::GetInstance()->set_jit_status(kNotJit); }
};

std::string GetJitLevel();

std::string GetObjDesc(const py::object &source);
bool IsPhaseLoadFromMindIR(const std::string &phase);

bool IsPhaseExport(const std::string &phase);
void SetLoopCount(const ResourcePtr &resource);
void ResetId(const ResourcePtr &resource);
#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name);
void RecordIR(const size_t action_index, const size_t action_size, const std::string &action_name,
              const FuncGraphPtr &graph, FuncGraphPtr *user_graph);
#endif
AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden = false);
void AddManagerForFuncGraphArgs(const ResourcePtr &resource, const ValuePtrList &arguments);
void CheckInterpretNodeLineInfos();
void SetHookForArgAbstract(const ResourcePtr &resource, const py::object &arg, abstract::AbstractBasePtr abs);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PIPELINE_H_
