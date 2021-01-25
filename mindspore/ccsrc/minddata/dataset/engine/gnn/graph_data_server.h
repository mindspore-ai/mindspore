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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVER_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#if !defined(_WIN32) && !defined(_WIN64)
#include "grpcpp/grpcpp.h"
#include "minddata/dataset/engine/gnn/graph_data_service_impl.h"
#include "minddata/dataset/engine/gnn/grpc_async_server.h"
#endif
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
namespace gnn {

class GraphDataImpl;

class GraphDataServer {
 public:
  enum ServerState { kGdsUninit = 0, kGdsInitializing, kGdsRunning, kGdsStopped };
  GraphDataServer(const std::string &dataset_file, int32_t num_workers, const std::string &hostname, int32_t port,
                  int32_t client_num, bool auto_shutdown);
  ~GraphDataServer() = default;

  Status Init();

  Status Stop();

  Status ClientRegister(int32_t pid);
  Status ClientUnRegister(int32_t pid);

  enum ServerState state() { return state_; }

  bool IsStopped() {
    if (state_ == kGdsStopped) {
      return true;
    } else {
      return false;
    }
  }

 private:
  void set_state(enum ServerState state) { state_ = state; }

  Status InitGraphDataImpl();
#if !defined(_WIN32) && !defined(_WIN64)
  Status StartAsyncRpcService();
#endif
  Status JudgeAutoShutdownServer();

  std::string dataset_file_;
  int32_t num_workers_;  // The number of worker threads
  int32_t client_num_;
  int32_t max_connected_client_num_;
  bool auto_shutdown_;
  enum ServerState state_;
  std::unique_ptr<TaskGroup> tg_;  // Class for worker management
  std::unique_ptr<GraphDataImpl> graph_data_impl_;
  std::unordered_set<int32_t> client_pid_;
  std::mutex mutex_;
#if !defined(_WIN32) && !defined(_WIN64)
  std::unique_ptr<GraphDataServiceImpl> service_impl_;
  std::unique_ptr<GrpcAsyncServer> async_server_;
#endif
};

#if !defined(_WIN32) && !defined(_WIN64)
class UntypedCall {
 public:
  virtual ~UntypedCall() {}

  virtual Status operator()() = 0;

  virtual bool JudgeFinish() = 0;
};

template <class ServiceImpl, class AsyncService, class RequestMessage, class ResponseMessage>
class CallData : public UntypedCall {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };
  using EnqueueFunction = void (AsyncService::*)(grpc::ServerContext *, RequestMessage *,
                                                 grpc::ServerAsyncResponseWriter<ResponseMessage> *,
                                                 grpc::CompletionQueue *, grpc::ServerCompletionQueue *, void *);
  using HandleRequestFunction = grpc::Status (ServiceImpl::*)(grpc::ServerContext *, const RequestMessage *,
                                                              ResponseMessage *);
  CallData(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq,
           EnqueueFunction enqueue_function, HandleRequestFunction handle_request_function)
      : status_(STATE::CREATE),
        service_impl_(service_impl),
        async_service_(async_service),
        cq_(cq),
        enqueue_function_(enqueue_function),
        handle_request_function_(handle_request_function),
        responder_(&ctx_) {}

  ~CallData() = default;

  static Status EnqueueRequest(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq,
                               EnqueueFunction enqueue_function, HandleRequestFunction handle_request_function) {
    auto call = new CallData<ServiceImpl, AsyncService, RequestMessage, ResponseMessage>(
      service_impl, async_service, cq, enqueue_function, handle_request_function);
    RETURN_IF_NOT_OK((*call)());
    return Status::OK();
  }

  Status operator()() override {
    if (status_ == STATE::CREATE) {
      status_ = STATE::PROCESS;
      (async_service_->*enqueue_function_)(&ctx_, &request_, &responder_, cq_, cq_, this);
    } else if (status_ == STATE::PROCESS) {
      EnqueueRequest(service_impl_, async_service_, cq_, enqueue_function_, handle_request_function_);
      status_ = STATE::FINISH;
      grpc::Status s = (service_impl_->*handle_request_function_)(&ctx_, &request_, &response_);
      responder_.Finish(response_, s, this);
    } else {
      MS_LOG(WARNING) << "The CallData status is finish and the pointer needs to be released.";
    }
    return Status::OK();
  }

  bool JudgeFinish() override {
    if (status_ == STATE::FINISH) {
      return true;
    } else {
      return false;
    }
  }

 private:
  STATE status_;
  ServiceImpl *service_impl_;
  AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  EnqueueFunction enqueue_function_;
  HandleRequestFunction handle_request_function_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
  RequestMessage request_;
  ResponseMessage response_;
};

#define ENQUEUE_REQUEST(service_impl, async_service, cq, method, request_msg, response_msg)                       \
  do {                                                                                                            \
    Status s =                                                                                                    \
      CallData<gnn::GraphDataServiceImpl, GnnGraphData::AsyncService, request_msg, response_msg>::EnqueueRequest( \
        service_impl, async_service, cq, &GnnGraphData::AsyncService::Request##method,                            \
        &gnn::GraphDataServiceImpl::method);                                                                      \
    RETURN_IF_NOT_OK(s);                                                                                          \
  } while (0)

class GraphDataGrpcServer : public GrpcAsyncServer {
 public:
  GraphDataGrpcServer(const std::string &host, int32_t port, GraphDataServiceImpl *service_impl)
      : GrpcAsyncServer(host, port), service_impl_(service_impl) {}

  ~GraphDataGrpcServer() = default;

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return Status::OK();
  }

  Status EnqueueRequest() {
    ENQUEUE_REQUEST(service_impl_, &svc_, cq_.get(), ClientRegister, GnnClientRegisterRequestPb,
                    GnnClientRegisterResponsePb);
    ENQUEUE_REQUEST(service_impl_, &svc_, cq_.get(), ClientUnRegister, GnnClientUnRegisterRequestPb,
                    GnnClientUnRegisterResponsePb);
    ENQUEUE_REQUEST(service_impl_, &svc_, cq_.get(), GetGraphData, GnnGraphDataRequestPb, GnnGraphDataResponsePb);
    ENQUEUE_REQUEST(service_impl_, &svc_, cq_.get(), GetMetaInfo, GnnMetaInfoRequestPb, GnnMetaInfoResponsePb);
    return Status::OK();
  }

  Status ProcessRequest(void *tag) {
    auto rq = static_cast<UntypedCall *>(tag);
    if (rq->JudgeFinish()) {
      delete rq;
    } else {
      RETURN_IF_NOT_OK((*rq)());
    }
    return Status::OK();
  }

 private:
  GraphDataServiceImpl *service_impl_;
  GnnGraphData::AsyncService svc_;
};
#endif
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVER_H_
