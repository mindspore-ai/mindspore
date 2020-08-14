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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRPC_ASYNC_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRPC_ASYNC_SERVER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/async_unary_call.h"

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

/// \brief Async server base class
class GrpcAsyncServer {
 public:
  explicit GrpcAsyncServer(const std::string &host, int32_t port);
  virtual ~GrpcAsyncServer();
  /// \brief Brings up gRPC server
  /// \return none
  Status Run();
  /// \brief Entry function to handle async server request
  Status HandleRequest();

  void Stop();

  virtual Status RegisterService(grpc::ServerBuilder *builder) = 0;

  virtual Status EnqueueRequest() = 0;

  virtual Status ProcessRequest(void *tag) = 0;

 protected:
  int32_t port_;
  std::string host_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRPC_ASYNC_SERVER_H_
