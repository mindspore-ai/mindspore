/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <mutex>

#include "pybind11/pybind11.h"

#include "ir/anf.h"
#include "ir/tensor.h"
#include "pipeline/jit/action.h"
#include "vm/segment_runner.h"
#include "vm/transform.h"
#include "pipeline/jit/base.h"

namespace mindspore {
extern const char kMsConvert[];
extern const char kMsVm[];

// namespace to support pipeline structures definition
namespace pipeline {

namespace py = pybind11;

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

// A function pipeline.
class ExecutorPy : public std::enable_shared_from_this<ExecutorPy> {
 public:
  static std::shared_ptr<ExecutorPy> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<ExecutorPy>(new (std::nothrow) ExecutorPy());
    }
    return executor_;
  }

  ~ExecutorPy();

  void SaveCompiledGraph(const std::string &phase_s);
  bool CompileInner(const py::object &obj, const py::tuple &args, const py::object &phase, bool use_vm);
  bool Compile(const py::object &obj, const py::tuple &args, const py::object &phase, bool use_vm);

  void ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *arg_list);

  // for pynative mode when use_vm is on
  py::object Run(const py::tuple &args, const py::object &phase);
  ResourcePtr GetResource(const std::string &phase);
  FuncGraphPtr GetFuncGraph(const std::string &phase);
  py::bytes GetFuncGraphProto(const std::string &phase, const std::string &type);
  compile::VmEvalFuncPtr GetVmEvalFunc(const std::string &phase);
  bool HasCompiled(const std::string &phase) const;

  FuncGraphPtr BuildGraph(const py::dict &init_params, const std::string &phase,
                          const py::object &broadcast_params = {});
  void UpdataParamNodeDefaultInput(const std::string &phase,
                                   const std::unordered_map<std::string, tensor::TensorPtr> &params);
  void RunInitGraph(const py::dict &init_params, const std::string &phase);
  void PyExePath(const py::object &phase);
  py::dict GetParameterLayout(const std::string &phase);
  py::dict GetCNodeStrategy(const std::string &phase);
  py::list GetParallelParameterNameList(const std::string &phase);
  void SetCNodeStrategy(const std::string &name, const parallel::Strategys &strategy);
  size_t GetNumOpsInfo(const std::string &phase);
  void SetNumOpsInfo(size_t);
  py::dict GetAllreduceFusion(const std::string &phase);
  void DelNetRes(const std::string &id);
  void ReleaseResource(const py::object &phase);
  static void ClearRes();
  static bool GetDebugTerminate() { return debugger_terminate_; }
  static void DebugTerminate(bool val) { debugger_terminate_ = val; }
  void TerminateDebugger();

  std::map<std::string, std::pair<PrimitivePyPtr, std::string>> FetchInfoForQuantExport(const std::string &phase_s);

 private:
  ExecutorPy();
  void ConvertObjectToTensors(const py::dict &dict, std::map<std::string, tensor::TensorPtr> *tensors);
  void GetWeightInfo(const CNodePtr &root_node, const AnfNodePtr &weight_node,
                     std::map<std::string, std::pair<PrimitivePyPtr, std::string>> *fake_quant_table);
  void GetGeBackendPolicy() const;
  // filter some pipeline actions according to phase, e.g. when exporting onnx, it is no need to execute actions after
  // 'validate' stage
  static std::vector<ActionItem> FilterActions(const std::vector<ActionItem> &actions, const std::string &phase);

  std::map<std::string, ExecutorInfoPtr> info_;
  static std::shared_ptr<ExecutorPy> executor_;
  static std::mutex instance_lock_;
  static bool debugger_terminate_;
  std::map<std::string, py::dict> stra_dict_;
  std::string phase_ = "";
  std::map<std::string, size_t> phase_to_num_op_info_;
};
using ExecutorPyPtr = std::shared_ptr<ExecutorPy>;

// Generate a key for mapping function graph
py::tuple GenerateKey(const std::string &name, const std::unordered_map<std::string, py::object> &defaults);
py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs);

bool InitDistribute(const std::map<std::string, std::string> &options);

void ResetOpId();
void InitHccl();
void FinalizeHccl();
void InitPipeline();
void FinalizeBackend();
void ClearResAtexit();
void ReleaseGeTsd();

void ExportGraph(const std::string &file_name, const std::string &, const std::string &phase);
FuncGraphPtr LoadMindIR(const std::string &file_name);

// init and exec dataset sub graph
bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run);

// Build and run dataset subgraph for ms backend
bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run);

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list);

}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_H_
