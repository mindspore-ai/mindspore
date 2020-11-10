/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/gnn/grpc_async_server.h"

#include <limits>
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {

GrpcAsyncServer::GrpcAsyncServer(const std::string &host, int32_t port) : host_(host), port_(port) {}

GrpcAsyncServer::~GrpcAsyncServer() { Stop(); }

Status GrpcAsyncServer::Run() {
  std::string server_address = host_ + ":" + std::to_string(port_);
  grpc::ServerBuilder builder;
  // Default message size for gRPC is 4MB. Increase it to 2g-1
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  int port_tcpip = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port_tcpip);
  RETURN_IF_NOT_OK(RegisterService(&builder));
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  if (server_) {
    MS_LOG(INFO) << "Server listening on " << server_address;
  } else {
    std::string errMsg = "Fail to start server. ";
    if (port_tcpip != port_) {
      errMsg += "Unable to bind to address " + server_address + ".";
    }
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status GrpcAsyncServer::HandleRequest() {
  bool success = false;
  void *tag;
  // We loop through the grpc queue. Each connection if successful
  // will come back with our own tag which is an instance of CallData
  // and we simply call its functor. But first we need to create these instances
  // and inject them into the grpc queue.
  RETURN_IF_NOT_OK(EnqueueRequest());
  while (cq_->Next(&tag, &success)) {
    RETURN_IF_INTERRUPTED();
    if (success) {
      RETURN_IF_NOT_OK(ProcessRequest(tag));
    } else {
      MS_LOG(DEBUG) << "cq_->Next failed.";
    }
  }
  return Status::OK();
}

void GrpcAsyncServer::Stop() {
  if (server_) {
    server_->Shutdown();
  }
  // Always shutdown the completion queue after the server.
  if (cq_) {
    cq_->Shutdown();
  }
}
}  // namespace dataset
}  // namespace mindspore
