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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_

#include <grpcpp/grpcpp.h>
#include <string>
#include <list>
#include <vector>
#include <memory>
#include "proto/debug_grpc.grpc.pb.h"

using debugger::Chunk;
using debugger::EventListener;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::Heartbeat;
using debugger::Metadata;
using debugger::TensorBase;
using debugger::TensorProto;
using debugger::TensorSummary;
using debugger::WatchpointHit;

namespace mindspore {
class GrpcClient {
 public:
  GrpcClient(const std::string &host, const std::string &port);

  ~GrpcClient() = default;

  void Init(const std::string &host, const std::string &port);

  void Reset();

  EventReply WaitForCommand(const Metadata &metadata);

  EventReply SendMetadata(const Metadata &metadata);

  EventReply SendGraph(const GraphProto &graph);

  EventReply SendTensors(const std::list<TensorProto> &tensors);

  EventReply SendTensorBase(const std::list<TensorBase> &tensor_base);

  EventReply SendTensorStats(const std::list<TensorSummary> &tensor_summary);

  EventReply SendMultiGraphs(const std::list<Chunk> &chunks);

  EventReply SendWatchpointHits(const std::list<WatchpointHit> &watchpoints);

  std::vector<std::string> ChunkString(std::string str, int graph_size);

  EventReply SendHeartbeat(const Heartbeat &heartbeat);

 private:
  std::unique_ptr<EventListener::Stub> stub_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_
