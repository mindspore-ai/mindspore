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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_

#include <grpcpp/grpcpp.h>

#include <stdio.h>
#include <stdlib.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/pkcs12.h>

#include <string>
#include <list>
#include <memory>

#include "proto/debug_grpc.grpc.pb.h"

using debugger::EventListener;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::Metadata;
using debugger::TensorProto;
using debugger::WatchpointHit;

namespace mindspore {
class GrpcClient {
 public:
  // constructor
  GrpcClient(const std::string &host, const std::string &port, const bool &ssl_certificate,
             const std::string &certificate_dir, const std::string &certificate_passphrase);

  // deconstructor
  ~GrpcClient() = default;

  // init
  void Init(const std::string &host, const std::string &port, const bool &ssl_certificate,
            const std::string &certificate_dir, const std::string &certificate_passphrase);

  // reset
  void Reset();

  EventReply WaitForCommand(const Metadata &metadata);

  EventReply SendMetadata(const Metadata &metadata);

  EventReply SendGraph(const GraphProto &graph);

  EventReply SendTensors(const std::list<TensorProto> &tensors);

  EventReply SendWatchpointHits(const std::list<WatchpointHit> &watchpoints);

 private:
  std::unique_ptr<EventListener::Stub> stub_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DEBUGGER_GRPC_CLIENT_H_
