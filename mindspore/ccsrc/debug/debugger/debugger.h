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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_

#include <list>
#include <memory>
#include <string>
#include "session/kernel_graph.h"
#include "debug/debugger/grpc_client.h"
#include "debug/debug_services.h"

using debugger::DataType;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::ModelProto;
using debugger::TensorProto;
using debugger::WatchCondition;
using debugger::WatchNode;
using debugger::WatchpointHit;

template <class T>
using ProtoVector = google::protobuf::RepeatedPtrField<T>;

namespace mindspore {
// different types of command recieved by debugger
// need to keep sync with client-side proto and server-side proto
enum class DebuggerCommand { kExitCMD = 2, kRunCMD = 3, kSetCMD = 4, kViewCMD = 5, kUnknownCMD = -1 };

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
  void Init(const uint32_t device_id);

  // reset debugger
  void Reset();

  // enable debugger
  // send graph and wait for command
  // do nothing if graph is set already
  void PreExecute(const KernelGraphPtr &graph_ptr);

  // analyze tensors and wait for command
  // don't need a graph_ptr because it is saved during pre_execute
  void PostExecute();

  // suspend the execution after a debug_op
  void PostDebugOp();

  DebugServices *get_debug_services();

  bool debugger_enabled();

 private:
  // private constructor for singleton
  Debugger();

  // enable debugger
  // instantiate class members
  // read env variable for grpc client
  void EnableDebugger();

  // check and save graph pointer
  void CheckGraphPtr(const KernelGraphPtr &graph_ptr);

  // check if the graph is a dataset graph
  void CheckDatasetGraph();

  // serialize graph and get proto
  GraphProto GetGraphProto();

  // send graph and enter command wait loop
  void SendGraphAndSuspend(const GraphProto &graph_proto);

  // wait for command and process command
  // send command request and process reply in a loop
  // break if RunCMD
  void CommandLoop();

  // process reply and command type
  DebuggerCommand GetCommand(const EventReply &reply);

  // parse other data out of EventReply
  ProtoVector<WatchNode> GetWatchnodes(const EventReply &reply);
  WatchCondition GetWatchcondition(const EventReply &reply);
  int32_t GetWatchpointID(const EventReply &reply);
  bool GetWatchpointDelete(const EventReply &reply);
  ProtoVector<TensorProto> GetTensors(const EventReply &reply);

  // set what nodes and conditions to watch
  void SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id);

  // remove watchpoint with id
  void RemoveWatchpoint(const int32_t id);

  // load tensor for view command
  std::list<TensorProto> LoadTensors(const ProtoVector<TensorProto> &tensors);

  // terminate training process
  void Exit();

  // analyze tensors and check watchpoint conditions
  // return names of tensors and what condition they hit
  std::list<WatchpointHit> CheckWatchpoints();

  // send watchpoints that hit and enter command wait loop
  void SendWatchpointsAndSuspend(const std::list<WatchpointHit> &points);

  // class members
  std::unique_ptr<GrpcClient> grpc_client_;
  std::unique_ptr<DebugServices> debug_services_;
  KernelGraphPtr graph_ptr_;
  uint32_t device_id_;
  int32_t num_step_;
  bool debugger_enabled_;
  bool is_dataset_graph_;
  std::mutex access_lock_;

  // singleton
  static std::mutex instance_lock_;
  static std::shared_ptr<Debugger> debugger_;
};

using DebuggerPtr = std::shared_ptr<Debugger>;

// get debugger ModelProto
std::string GetDebuggerFuncGraphProtoString(const FuncGraphPtr &func_graph);
ModelProto GetDebuggerFuncGraphProto(const FuncGraphPtr &func_graph);

// for getting proto DataType from Type of Tensor
DataType GetDebuggerNumberDataType(const TypePtr &type);

}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_H_
