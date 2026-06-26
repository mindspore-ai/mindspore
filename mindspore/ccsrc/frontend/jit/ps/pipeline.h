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
#include "include/common/visible.h"
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
    UCEException::GetInstance().SetGraphPipelineCompiled(true);
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
FRONTEND_EXPORT void CheckArgsValid(const py::object &source, const py::tuple &args);
FRONTEND_EXPORT py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs);

bool InitDistribute(const std::map<std::string, std::string> &options);

FRONTEND_EXPORT void ResetOpId();
FRONTEND_EXPORT void ResetOpIdWithOffset();
FRONTEND_EXPORT void InitHccl();
FRONTEND_EXPORT void InitHccl(int64_t timeout, uint32_t world_size, uint32_t node_id,
                              distributed::cluster::TCPStoreClientPtr store, std::string group_name);
FRONTEND_EXPORT void FinalizeHccl();
FRONTEND_EXPORT uint32_t GetHcclRankId();
FRONTEND_EXPORT uint32_t GetHcclRankSize();
FRONTEND_EXPORT void InitPipeline();

FRONTEND_EXPORT void BindDeviceCtx();

FRONTEND_EXPORT FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                                        const std::string &dec_mode, const py::object decrypt = py::none());

FRONTEND_EXPORT FuncGraphPtr SplitMindIR(const std::string &file_name);

FRONTEND_EXPORT FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id,
                                                bool sapp);

// init and exec dataset sub graph
bool FRONTEND_EXPORT InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                                     const std::vector<int64_t> &input_indexes, const std::string &phase,
                                     bool need_run);

// Build and run dataset subgraph for ms backend
bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run);

FRONTEND_EXPORT py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len,
                                    const std::string &enc_mode);
FRONTEND_EXPORT py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len,
                                    const std::string &dec_mode);
FRONTEND_EXPORT py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len,
                                        const std::string &dec_mode);
FRONTEND_EXPORT bool PyIsCipherFile(const std::string &file_path);
FRONTEND_EXPORT void FinalizeCluster();
FRONTEND_EXPORT void SwapCache(const py::object &host_, const py::object &device_, const py::object &block_mapping_,
                               const bool &type);

bool IsPhaseExport(const std::string &phase);
py::object BaseRefToPyDataWithUserData(const BaseRef &value, const AbstractBasePtr &abs);
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
FRONTEND_EXPORT bool RunJitPipeline();
FRONTEND_EXPORT std::string DumpFuncGraph(const py::object &obj);
FRONTEND_EXPORT void PreJit(const py::object &args, const py::object &kwargs);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PIPELINE_H_
