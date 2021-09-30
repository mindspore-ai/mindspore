/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "backend/session/kernel_graph.h"
#include "debug/debugger/grpc_client.h"
#include "debug/debug_services.h"
#include "common/trans.h"

using debugger::Chunk;
using debugger::DataType;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::ModelProto;
using debugger::Statistics;
using debugger::TensorProto;
using debugger::WatchCondition;
using debugger::WatchCondition_Parameter;
using debugger::WatchNode;
using debugger::WatchpointHit;

template <class T>
using ProtoVector = google::protobuf::RepeatedPtrField<T>;

namespace mindspore {
// different types of command received by debugger
// need to keep sync with client-side proto and server-side proto
enum class DebuggerCommand {
  kExitCMD = 2,
  kRunCMD = 3,
  kSetCMD = 4,
  kViewCMD = 5,
  kVersionMatchedCMD = 6,
  kUnknownCMD = -1
};

class Debugger : public std::enable_shared_from_this<Debugger> {
 public:
  static std::shared_ptr<Debugger> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (debugger_ == nullptr) {
      debugger_ = std::shared_ptr<Debugger>(new (std::nothrow) Debugger());
    }
    return debugger_;
  }

  // deconstructor
  ~Debugger() = default;

  // init
  // only save device_id
  void Init(const uint32_t device_id, const std::string device_target);

  // reset debugger
  void Reset();

  void PreExecuteGraphDebugger(const std::vector<KernelGraphPtr> &graphs);
  // enable debugger
  // send graph and wait for command
  // do nothing if graph is set already
  void PreExecute(const KernelGraphPtr &graph_ptr);

  // analyze tensors and wait for command
  // don't need a graph_ptr because it is saved during pre_execute
  void PostExecute();

  bool DumpDataEnabledIteration() const;

  static uint32_t GetRankID();

  void Dump(const KernelGraphPtr &kernel_graph) const;

  void DumpSingleNode(const CNodePtr &node, uint32_t graph_id);

  void DumpSetup(const KernelGraphPtr &kernel_graph) const;

  void DumpInGraphCompiler(const KernelGraphPtr &kernel_graph);

  void PostExecuteGraphDebugger();

  bool ReadNodeDataRequired(const CNodePtr &kernel) const;

  void PostExecuteNode(const CNodePtr &kernel, bool last_kernel);

  bool DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                        const std::string &host_fmt, const std::vector<int64_t> &host_shape, TypeId host_type,
                        TypeId device_type, const std::string &addr_format, size_t slot) const;

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  bool debugger_enabled() const;

  bool partial_memory() const;

  void SetEnableHeartbeat(bool enabled);

  void SetCurNode(const std::string &cur_name);

  std::string run_level() const;

  // check if any feature that uses the debugger backend is enabled
  bool DebuggerBackendEnabled() const;

  void SetTrainingDone(bool training_done);

  // returns true if reply received and mindspore version matched with mindinsight version
  // version_check should be true if you want the function to do backend compatibility check with Mindinsight
  bool SendMetadata(bool version_check);

  void LoadParametersAndConst();

  void LoadParametersAndConst(const KernelGraphPtr &graph);

  void UpdateStepNum(const session::KernelGraph *graph);

  void UpdateStepNumGPU();

  void ClearCurrentData();

  void LoadGraphOutputs();

  void CheckDatasetSinkMode();

  void LoadGraphs(const KernelGraphPtr &graph_ptr);

  uint32_t GetFirstRunGraphId() const;

  void SetGraphPtr(const KernelGraphPtr &graph_ptr) { graph_ptr_ = graph_ptr; }

  const KernelGraphPtr GetGraphPtr() const { return graph_ptr_; }

  const std::list<KernelGraphPtr> GetGraphPtrList() const { return graph_ptr_list_; }

  bool TensorExistsInCurrent(const std::string &tensor_name);

  // check if dump using debugger backend is enabled
  bool CheckDebuggerDumpEnabled() const;

 private:
  // private constructor for singleton
  Debugger();

  // enable debugger
  // instantiate class members
  // read env variable for grpc client
  void EnableDebugger();

  // check if debugger enabled
  bool CheckDebuggerEnabled() const;

  void CheckDebuggerEnabledParam() const;

  bool CheckDebuggerPartialMemoryEnabled() const;

  // check and save graph pointer
  void CheckGraphPtr(const KernelGraphPtr &graph_ptr);

  // check if the graph is a dataset graph
  void CheckDatasetGraph();

  // serialize graph and get proto
  GraphProto GetGraphProto(const KernelGraphPtr &graph_ptr) const;

  // send heartbeat message to UI once per 30 second by default
  void SendHeartbeat(int32_t period);

  // send graph and enter command wait loop
  void SendGraphAndSuspend(const GraphProto &graph_proto);

  void SendMultiGraphsAndSuspend(const std::list<GraphProto> &graph_proto_list);

  // send multi_graphs and clear the graph_proto_list_
  void SendMultiGraphsAndClear(const KernelGraphPtr &graph_ptr);

  // wait for command and process command
  // send command request and process reply in a loop
  // break if RunCMD
  void CommandLoop();

  // Process the RunCMD
  void ProcessRunCMD(const EventReply &reply);
  // Process the KSetCMD
  void ProcessKSetCMD(const EventReply &reply);
  // Process the KViewCMD
  void ProcessKViewCMD(const EventReply &reply);
  // ViewCMD base level
  void ViewBaseLevel(const EventReply &reply);
  // ViewCMD statistics level
  void ViewStatLevel(const EventReply &reply);
  // ViewCMD value level
  void ViewValueLevel(const EventReply &reply);
  // set what nodes and conditions to watch
  void SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id,
                     const ProtoVector<WatchCondition_Parameter> &parameters);

  // remove watchpoint with id
  void RemoveWatchpoint(const int32_t id);

  // load tensor for view command
  std::list<TensorProto> LoadTensors(const ProtoVector<TensorProto> &tensors) const;

  // load tensor base for view command
  std::list<TensorBase> LoadTensorsBase(const ProtoVector<TensorProto> &tensors) const;

  // load tensor statistics for view command
  std::list<TensorSummary> LoadTensorsStat(const ProtoVector<TensorProto> &tensors) const;

  // terminate training process
  void Exit(bool exit_success = false);

  // analyze tensors and check watchpoint conditions
  // return names of tensors and what condition they hit
  std::list<WatchpointHit> CheckWatchpoints(const std::string &watchnode = std::string(),
                                            const CNodePtr &kernel = nullptr, bool recheck = false);

  // send watchpoints that hit
  void SendWatchpoints(const std::list<WatchpointHit> &points);

  // Check if the port is valid
  bool CheckPort(const std::string &port) const;

  // Check if the IP is valid
  bool CheckIp(const std::string &host) const;

  void LoadSingleAnfnode(const AnfNodePtr &anf_node, const size_t output_index);

  // class members

  std::unique_ptr<GrpcClient> grpc_client_;
  std::unique_ptr<DebugServices> debug_services_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  KernelGraphPtr graph_ptr_;
  uint32_t device_id_;
  std::string device_target_;
  int32_t num_step_;
  bool debugger_enabled_;
  bool suspended_at_last_kernel_;
  std::string run_level_;
  std::string node_name_;
  std::string cur_name_;
  bool training_done_;
  bool is_dataset_graph_;
  bool partial_memory_;
  std::mutex access_lock_;
  // flag to keep track of the very first suspension of debugger
  bool initial_suspend_;
  bool enable_heartbeat_;

  std::list<GraphProto> graph_proto_list_;
  std::list<KernelGraphPtr> graph_ptr_list_;
  // The vector of graph pointers that have been run in the current step.
  std::vector<KernelGraphPtr> graph_ptr_step_vec_;

  // singleton
  static std::mutex instance_lock_;
  static std::shared_ptr<Debugger> debugger_;
  uint32_t not_dataset_graph_sum_;
  std::list<uint32_t> rungraph_id_list_;
  std::string version_;
};

using DebuggerPtr = std::shared_ptr<Debugger>;
// get debugger ModelProto
ModelProto GetDebuggerFuncGraphProto(const FuncGraphPtr &func_graph);

// for getting proto DataType from Type of Tensor
DataType GetDebuggerNumberDataType(const TypePtr &type);

// process reply and command type
DebuggerCommand GetCommand(const EventReply &reply);

// parse other data out of EventReply
ProtoVector<WatchCondition_Parameter> GetParameters(const EventReply &reply);
ProtoVector<WatchNode> GetWatchnodes(const EventReply &reply);
std::string GetNodeName(const EventReply &reply);
std::string GetRunLevel(const EventReply &reply);
WatchCondition GetWatchcondition(const EventReply &reply);
int32_t GetWatchpointID(const EventReply &reply);
bool GetWatchpointDelete(const EventReply &reply);
ProtoVector<TensorProto> GetTensors(const EventReply &reply);
bool GetMiVersionMatched(const EventReply &reply);
// get the full name of a tensor, which is the name used in TensorLoader
std::string GetTensorFullName(const TensorProto &tensor);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_
