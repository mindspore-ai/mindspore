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
#include "debug/debugger/grpc_client.h"

#include <thread>
#include <vector>
#include "utils/log_adapter.h"

using debugger::Chunk;
using debugger::EventListener;
using debugger::EventReply;
using debugger::EventReply_Status_FAILED;
using debugger::GraphProto;
using debugger::Heartbeat;
using debugger::Metadata;
using debugger::TensorBase;
using debugger::TensorProto;
using debugger::TensorSummary;
using debugger::WatchpointHit;

namespace mindspore {
GrpcClient::GrpcClient(const std::string &host, const std::string &port) : stub_(nullptr) { Init(host, port); }

void GrpcClient::Init(const std::string &host, const std::string &port) {
  std::string target_str = host + ":" + port;
  MS_LOG(INFO) << "GrpcClient connecting to: " << target_str;

  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials());
  stub_ = EventListener::NewStub(channel);
}

void GrpcClient::Reset() { stub_ = nullptr; }

EventReply GrpcClient::WaitForCommand(const Metadata &metadata) {
  EventReply reply;
  grpc::ClientContext context;
  grpc::Status status = stub_->WaitCMD(&context, metadata, &reply);
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: WaitForCommand";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendMetadata(const Metadata &metadata) {
  EventReply reply;
  grpc::ClientContext context;
  grpc::Status status = stub_->SendMetadata(&context, metadata, &reply);
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendMetadata";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

std::vector<std::string> GrpcClient::ChunkString(std::string str, int graph_size) {
  std::vector<std::string> buf;
  constexpr auto l_chunk_size = 1024 * 1024 * 3;
  int size_iter = 0;
  while (size_iter < graph_size) {
    int chunk_size = l_chunk_size;
    if (graph_size - size_iter < l_chunk_size) {
      chunk_size = graph_size - size_iter;
    }
    std::string buffer;
    buffer.resize(chunk_size);
    auto err = memcpy_s(reinterpret_cast<char *>(buffer.data()), chunk_size, str.data() + size_iter, chunk_size);
    if (err != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s failed. errorno is: " << err;
    }
    buf.push_back(buffer);
    if (size_iter > INT_MAX - l_chunk_size) {
      MS_EXCEPTION(ValueError) << size_iter << " + " << l_chunk_size << "would lead to integer overflow!";
    }
    size_iter += l_chunk_size;
  }
  return buf;
}

EventReply GrpcClient::SendGraph(const GraphProto &graph) {
  EventReply reply;
  grpc::ClientContext context;
  Chunk chunk;

  std::unique_ptr<grpc::ClientWriter<Chunk> > writer(stub_->SendGraph(&context, &reply));
  std::string str = graph.SerializeAsString();
  int graph_size = graph.ByteSize();
  auto buf = ChunkString(str, graph_size);

  for (unsigned int i = 0; i < buf.size(); i++) {
    MS_LOG(INFO) << "RPC:sending the " << i << "chunk in graph";
    chunk.set_buffer(buf[i]);
    if (!writer->Write(chunk)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendGraph";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendMultiGraphs(const std::list<Chunk> &chunks) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<Chunk> > writer(stub_->SendMultiGraphs(&context, &reply));
  for (const auto &chunk : chunks) {
    if (!writer->Write(chunk)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendMultigraphs";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendTensors(const std::list<TensorProto> &tensors) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<TensorProto> > writer(stub_->SendTensors(&context, &reply));
  for (const auto &tensor : tensors) {
    if (!writer->Write(tensor)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendTensors";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendWatchpointHits(const std::list<WatchpointHit> &watchpoints) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<WatchpointHit> > writer(stub_->SendWatchpointHits(&context, &reply));
  for (const auto &watchpoint : watchpoints) {
    if (!writer->Write(watchpoint)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendWatchpointHits";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendHeartbeat(const Heartbeat &heartbeat) {
  EventReply reply;
  grpc::ClientContext context;

  grpc::Status status = stub_->SendHeartbeat(&context, heartbeat, &reply);
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendHeartbeat";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendTensorBase(const std::list<TensorBase> &tensor_base_list) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<TensorBase> > writer(stub_->SendTensorBase(&context, &reply));

  for (const auto &tensor_base : tensor_base_list) {
    if (!writer->Write(tensor_base)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendTensorBase";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendTensorStats(const std::list<TensorSummary> &tensor_summary_list) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<TensorSummary> > writer(stub_->SendTensorStats(&context, &reply));

  for (const auto &tensor_summary : tensor_summary_list) {
    if (!writer->Write(tensor_summary)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();
  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendTensorStats";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}
}  // namespace mindspore
