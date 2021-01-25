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
#include "minddata/dataset/engine/gnn/graph_data_client.h"

#include <unistd.h>
#include <functional>
#include <map>

#if !defined(_WIN32) && !defined(_WIN64)
#include "grpcpp/grpcpp.h"
#endif

#include "minddata/dataset/core/data_type.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/tensor_proto.h"
#endif

namespace mindspore {
namespace dataset {
namespace gnn {

GraphDataClient::GraphDataClient(const std::string &dataset_file, const std::string &hostname, int32_t port)
    : dataset_file_(dataset_file),
      host_(hostname),
      port_(port),
      pid_(0),
#if !defined(_WIN32) && !defined(_WIN64)
      shared_memory_key_(-1),
      shared_memory_size_(0),
      graph_feature_parser_(nullptr),
      graph_shared_memory_(nullptr),
#endif
      registered_(false) {
}

GraphDataClient::~GraphDataClient() { (void)Stop(); }

Status GraphDataClient::Init() {
#if defined(_WIN32) || defined(_WIN64)
  RETURN_STATUS_UNEXPECTED("Graph data client is not supported in Windows OS");
#else
  if (!registered_) {
    std::string server_address;
    server_address = host_ + ":" + std::to_string(port_);
    MS_LOG(INFO) << "Graph data client starting. address:" << server_address;
    pid_ = getpid();
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(-1);
    std::shared_ptr<grpc::Channel> channel =
      grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args);
    stub_ = GnnGraphData::NewStub(channel);
    Status status = RegisterToServer();
    while (status.ToString().find("Initializing") != std::string::npos) {
      MS_LOG(INFO) << "Graph data server is initializing, please wait.";
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      status = RegisterToServer();
    }
    RETURN_IF_NOT_OK(status);
    MS_LOG(INFO) << "Graph data client successfully registered with server " << server_address;
  }
  RETURN_IF_NOT_OK(InitFeatureParser());
  return Status::OK();
#endif
}

Status GraphDataClient::Stop() {
#if !defined(_WIN32) && !defined(_WIN64)
  if (registered_) {
    UnRegisterToServer();
  }
#endif
  return Status::OK();
}

Status GraphDataClient::GetAllNodes(NodeType node_type, std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_ALL_NODES);
  request.add_type(static_cast<google::protobuf::int32>(node_type));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetAllEdges(EdgeType edge_type, std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_ALL_EDGES);
  request.add_type(static_cast<google::protobuf::int32>(edge_type));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetNodesFromEdges(const std::vector<EdgeIdType> &edge_list, std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_NODES_FROM_EDGES);
  for (const auto &edge_id : edge_list) {
    request.add_id(static_cast<google::protobuf::int32>(edge_id));
  }
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                                        std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_ALL_NEIGHBORS);
  for (const auto &node_id : node_list) {
    request.add_id(static_cast<google::protobuf::int32>(node_id));
  }
  request.add_type(static_cast<google::protobuf::int32>(neighbor_type));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetSampledNeighbors(const std::vector<NodeIdType> &node_list,
                                            const std::vector<NodeIdType> &neighbor_nums,
                                            const std::vector<NodeType> &neighbor_types, SamplingStrategy strategy,
                                            std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_SAMPLED_NEIGHBORS);
  for (const auto &node_id : node_list) {
    request.add_id(static_cast<google::protobuf::int32>(node_id));
  }
  for (const auto &num : neighbor_nums) {
    request.add_number(static_cast<google::protobuf::int32>(num));
  }
  for (const auto &type : neighbor_types) {
    request.add_type(static_cast<google::protobuf::int32>(type));
  }
  request.set_strategy(static_cast<google::protobuf::int32>(strategy));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetNegSampledNeighbors(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                               NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_NEG_SAMPLED_NEIGHBORS);
  for (const auto &node_id : node_list) {
    request.add_id(static_cast<google::protobuf::int32>(node_id));
  }
  request.add_number(static_cast<google::protobuf::int32>(samples_num));
  request.add_type(static_cast<google::protobuf::int32>(neg_neighbor_type));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GraphDataClient::RandomWalk(const std::vector<NodeIdType> &node_list,
                                                    const std::vector<NodeType> &meta_path, float step_home_param,
                                                    float step_away_param, NodeIdType default_node,
                                                    std::shared_ptr<Tensor> *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(RANDOM_WALK);
  for (const auto &node_id : node_list) {
    request.add_id(static_cast<google::protobuf::int32>(node_id));
  }
  for (const auto &type : meta_path) {
    request.add_type(static_cast<google::protobuf::int32>(type));
  }
  auto walk_param = request.mutable_random_walk();
  walk_param->set_p(step_home_param);
  walk_param->set_q(step_away_param);
  walk_param->set_default_id(static_cast<google::protobuf::int32>(default_node));
  RETURN_IF_NOT_OK(GetGraphDataTensor(request, &response, out));
#endif
  return Status::OK();
}

Status GraphDataClient::GetNodeFeature(const std::shared_ptr<Tensor> &nodes,
                                       const std::vector<FeatureType> &feature_types, TensorRow *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (!nodes || nodes->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input nodes is empty");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!feature_types.empty(), "Input feature_types is empty");

  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_NODE_FEATURE);
  for (const auto &type : feature_types) {
    request.add_type(static_cast<google::protobuf::int32>(type));
  }
  RETURN_IF_NOT_OK(TensorToPb(nodes, request.mutable_id_tensor()));
  RETURN_IF_NOT_OK(GetGraphData(request, &response));
  CHECK_FAIL_RETURN_UNEXPECTED(feature_types.size() == response.result_data().size(),
                               "The number of feature types returned by the server is wrong");
  if (response.result_data().size() > 0) {
    size_t i = 0;
    for (const auto &result : response.result_data()) {
      std::shared_ptr<Tensor> tensor;
      RETURN_IF_NOT_OK(PbToTensor(&result, &tensor));
      std::shared_ptr<Tensor> fea_tensor;
      RETURN_IF_NOT_OK(ParseNodeFeatureFromMemory(nodes, feature_types[i], tensor, &fea_tensor));
      out->emplace_back(std::move(fea_tensor));
      ++i;
    }
  } else {
    RETURN_STATUS_UNEXPECTED("RPC failed: The number of returned tensor is abnormal");
  }
#endif
  return Status::OK();
}

Status GraphDataClient::GetEdgeFeature(const std::shared_ptr<Tensor> &edges,
                                       const std::vector<FeatureType> &feature_types, TensorRow *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (!edges || edges->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input edges is empty");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!feature_types.empty(), "Input feature_types is empty");

  GnnGraphDataRequestPb request;
  GnnGraphDataResponsePb response;
  request.set_op_name(GET_EDGE_FEATURE);
  for (const auto &type : feature_types) {
    request.add_type(static_cast<google::protobuf::int32>(type));
  }
  RETURN_IF_NOT_OK(TensorToPb(edges, request.mutable_id_tensor()));
  RETURN_IF_NOT_OK(GetGraphData(request, &response));
  CHECK_FAIL_RETURN_UNEXPECTED(feature_types.size() == response.result_data().size(),
                               "The number of feature types returned by the server is wrong");
  if (response.result_data().size() > 0) {
    size_t i = 0;
    for (const auto &result : response.result_data()) {
      std::shared_ptr<Tensor> tensor;
      RETURN_IF_NOT_OK(PbToTensor(&result, &tensor));
      std::shared_ptr<Tensor> fea_tensor;
      RETURN_IF_NOT_OK(ParseEdgeFeatureFromMemory(edges, feature_types[i], tensor, &fea_tensor));
      out->emplace_back(std::move(fea_tensor));
      ++i;
    }
  } else {
    RETURN_STATUS_UNEXPECTED("RPC failed: The number of returned tensor is abnormal");
  }
#endif
  return Status::OK();
}

Status GraphDataClient::GraphInfo(py::dict *out) {
#if !defined(_WIN32) && !defined(_WIN64)
  RETURN_IF_NOT_OK(CheckPid());
  void *tag;
  bool ok;
  grpc::Status status;
  grpc::ClientContext ctx;
  grpc::CompletionQueue cq;
  GnnMetaInfoRequestPb request;
  GnnMetaInfoResponsePb response;
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  ctx.set_deadline(deadline);
  std::unique_ptr<grpc::ClientAsyncResponseReader<GnnMetaInfoResponsePb>> rpc(
    stub_->PrepareAsyncGetMetaInfo(&ctx, request, &cq));
  rpc->StartCall();
  rpc->Finish(&response, &status, &response);

  {
    py::gil_scoped_release gil_release;
    auto success = cq.Next(&tag, &ok);
    CHECK_FAIL_RETURN_UNEXPECTED(success, "Expect successful");
    CHECK_FAIL_RETURN_UNEXPECTED(tag == &response, "Expect the same tag");
    CHECK_FAIL_RETURN_UNEXPECTED(ok, "Expect successful");
  }

  if (status.ok()) {
    if (response.error_msg() != "Success") {
      RETURN_STATUS_UNEXPECTED(response.error_msg());
    } else {
      MetaInfo meta_info;
      for (const auto &node : response.node_info()) {
        meta_info.node_type.emplace_back(static_cast<NodeType>(node.type()));
        meta_info.node_num[static_cast<NodeType>(node.type())] = static_cast<NodeIdType>(node.num());
      }
      for (const auto &edge : response.edge_info()) {
        meta_info.edge_type.emplace_back(static_cast<EdgeType>(edge.type()));
        meta_info.edge_num[static_cast<EdgeType>(edge.type())] = static_cast<EdgeIdType>(edge.num());
      }
      for (const auto &feature_type : response.node_feature_type()) {
        meta_info.node_feature_type.emplace_back(static_cast<FeatureType>(feature_type));
      }
      for (const auto &feature_type : response.edge_feature_type()) {
        meta_info.edge_feature_type.emplace_back(static_cast<FeatureType>(feature_type));
      }
      (*out)["node_type"] = py::cast(meta_info.node_type);
      (*out)["edge_type"] = py::cast(meta_info.edge_type);
      (*out)["node_num"] = py::cast(meta_info.node_num);
      (*out)["edge_num"] = py::cast(meta_info.edge_num);
      (*out)["node_feature_type"] = py::cast(meta_info.node_feature_type);
      (*out)["edge_feature_type"] = py::cast(meta_info.edge_feature_type);
    }
  } else {
    auto error_code = status.error_code();
    RETURN_STATUS_UNEXPECTED(status.error_message() + ". GRPC Code " + std::to_string(error_code));
  }
#endif
  return Status::OK();
}

#if !defined(_WIN32) && !defined(_WIN64)
Status GraphDataClient::GetGraphData(const GnnGraphDataRequestPb &request, GnnGraphDataResponsePb *response) {
  RETURN_IF_NOT_OK(CheckPid());
  void *tag;
  bool ok;
  grpc::Status status;
  grpc::ClientContext ctx;
  grpc::CompletionQueue cq;
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  ctx.set_deadline(deadline);
  std::unique_ptr<grpc::ClientAsyncResponseReader<GnnGraphDataResponsePb>> rpc(
    stub_->PrepareAsyncGetGraphData(&ctx, request, &cq));
  rpc->StartCall();
  rpc->Finish(response, &status, response);

  {
    py::gil_scoped_release gil_release;
    auto success = cq.Next(&tag, &ok);
    CHECK_FAIL_RETURN_UNEXPECTED(success, "Expect successful");
    CHECK_FAIL_RETURN_UNEXPECTED(tag == response, "Expect the same tag");
    CHECK_FAIL_RETURN_UNEXPECTED(ok, "Expect successful");
  }

  if (status.ok()) {
    if (response->error_msg() != "Success") {
      RETURN_STATUS_UNEXPECTED(response->error_msg());
    }
  } else {
    auto error_code = status.error_code();
    RETURN_STATUS_UNEXPECTED(status.error_message() + ". GRPC Code " + std::to_string(error_code));
  }

  return Status::OK();
}

Status GraphDataClient::GetGraphDataTensor(const GnnGraphDataRequestPb &request, GnnGraphDataResponsePb *response,
                                           std::shared_ptr<Tensor> *out) {
  RETURN_IF_NOT_OK(GetGraphData(request, response));
  if (1 == response->result_data().size()) {
    const TensorPb &result = response->result_data()[0];
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(PbToTensor(&result, &tensor));
    *out = std::move(tensor);
  } else {
    RETURN_STATUS_UNEXPECTED("RPC failed: The number of returned tensor is abnormal");
  }
  return Status::OK();
}

Status GraphDataClient::ParseNodeFeatureFromMemory(const std::shared_ptr<Tensor> &nodes, FeatureType feature_type,
                                                   const std::shared_ptr<Tensor> &memory_tensor,
                                                   std::shared_ptr<Tensor> *out) {
  std::shared_ptr<Tensor> default_feature;
  // If no feature can be obtained, fill in the default value
  RETURN_IF_NOT_OK(GetNodeDefaultFeature(feature_type, &default_feature));
  TensorShape shape(default_feature->shape());
  auto shape_vec = nodes->shape().AsVector();
  dsize_t size = std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<dsize_t>());
  shape = shape.PrependDim(size);
  std::shared_ptr<Tensor> fea_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, default_feature->type(), &fea_tensor));

  dsize_t index = 0;
  auto fea_addr_itr = memory_tensor->begin<int64_t>();
  for (auto node_itr = nodes->begin<NodeIdType>(); node_itr != nodes->end<NodeIdType>(); ++node_itr) {
    int64_t offset = *fea_addr_itr;
    fea_addr_itr++;
    int64_t len = *fea_addr_itr;
    fea_addr_itr++;
    if (*node_itr == kDefaultNodeId || offset < 0 || len <= 0) {
      RETURN_IF_NOT_OK(fea_tensor->InsertTensor({index}, default_feature));
    } else {
      uchar *start_addr_of_index = nullptr;
      TensorShape remaining({-1});
      RETURN_IF_NOT_OK(fea_tensor->StartAddrOfIndex({index}, &start_addr_of_index, &remaining));
      RETURN_IF_NOT_OK(graph_shared_memory_->GetData(start_addr_of_index, len, offset, len));
    }
    index++;
  }

  TensorShape reshape(nodes->shape());
  for (auto s : default_feature->shape().AsVector()) {
    reshape = reshape.AppendDim(s);
  }
  RETURN_IF_NOT_OK(fea_tensor->Reshape(reshape));
  fea_tensor->Squeeze();

  *out = std::move(fea_tensor);
  return Status::OK();
}

Status GraphDataClient::ParseEdgeFeatureFromMemory(const std::shared_ptr<Tensor> &edges, FeatureType feature_type,
                                                   const std::shared_ptr<Tensor> &memory_tensor,
                                                   std::shared_ptr<Tensor> *out) {
  std::shared_ptr<Tensor> default_feature;
  // If no feature can be obtained, fill in the default value
  RETURN_IF_NOT_OK(GetEdgeDefaultFeature(feature_type, &default_feature));
  TensorShape shape(default_feature->shape());
  auto shape_vec = edges->shape().AsVector();
  dsize_t size = std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<dsize_t>());
  shape = shape.PrependDim(size);
  std::shared_ptr<Tensor> fea_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, default_feature->type(), &fea_tensor));

  dsize_t index = 0;
  auto fea_addr_itr = memory_tensor->begin<int64_t>();
  for (auto edge_itr = edges->begin<EdgeIdType>(); edge_itr != edges->end<EdgeIdType>(); ++edge_itr) {
    int64_t offset = *fea_addr_itr;
    fea_addr_itr++;
    int64_t len = *fea_addr_itr;
    fea_addr_itr++;
    if (offset < 0 || len <= 0) {
      RETURN_IF_NOT_OK(fea_tensor->InsertTensor({index}, default_feature));
    } else {
      uchar *start_addr_of_index = nullptr;
      TensorShape remaining({-1});
      RETURN_IF_NOT_OK(fea_tensor->StartAddrOfIndex({index}, &start_addr_of_index, &remaining));
      RETURN_IF_NOT_OK(graph_shared_memory_->GetData(start_addr_of_index, len, offset, len));
    }
    index++;
  }

  TensorShape reshape(edges->shape());
  for (auto s : default_feature->shape().AsVector()) {
    reshape = reshape.AppendDim(s);
  }
  RETURN_IF_NOT_OK(fea_tensor->Reshape(reshape));
  fea_tensor->Squeeze();

  *out = std::move(fea_tensor);
  return Status::OK();
}

Status GraphDataClient::GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Tensor> *out_feature) {
  auto itr = default_node_feature_map_.find(feature_type);
  if (itr == default_node_feature_map_.end()) {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *out_feature = itr->second;
  }
  return Status::OK();
}

Status GraphDataClient::GetEdgeDefaultFeature(FeatureType feature_type, std::shared_ptr<Tensor> *out_feature) {
  auto itr = default_edge_feature_map_.find(feature_type);
  if (itr == default_edge_feature_map_.end()) {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *out_feature = itr->second;
  }
  return Status::OK();
}

Status GraphDataClient::RegisterToServer() {
  RETURN_IF_NOT_OK(CheckPid());
  void *tag;
  bool ok;
  grpc::Status status;
  grpc::ClientContext ctx;
  grpc::CompletionQueue cq;
  GnnClientRegisterRequestPb request;
  GnnClientRegisterResponsePb response;
  request.set_pid(static_cast<google::protobuf::int32>(pid_));
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  ctx.set_deadline(deadline);
  std::unique_ptr<grpc::ClientAsyncResponseReader<GnnClientRegisterResponsePb>> rpc(
    stub_->PrepareAsyncClientRegister(&ctx, request, &cq));
  rpc->StartCall();
  rpc->Finish(&response, &status, &response);

  {
    py::gil_scoped_release gil_release;
    auto success = cq.Next(&tag, &ok);
    CHECK_FAIL_RETURN_UNEXPECTED(success, "Expect successful");
    CHECK_FAIL_RETURN_UNEXPECTED(tag == &response, "Expect the same tag");
    CHECK_FAIL_RETURN_UNEXPECTED(ok, "Expect successful");
  }

  if (status.ok()) {
    if (response.error_msg() == "Success") {
      registered_ = true;
      data_schema_ = mindrecord::json::parse(response.data_schema());
      shared_memory_key_ = static_cast<key_t>(response.shared_memory_key());
      shared_memory_size_ = response.shared_memory_size();
      MS_LOG(INFO) << "Register success, recv data_schema:" << response.data_schema();
      for (auto feature_info : response.default_node_feature()) {
        std::shared_ptr<Tensor> tensor;
        RETURN_IF_NOT_OK(PbToTensor(&feature_info.feature(), &tensor));
        default_node_feature_map_[feature_info.type()] = tensor;
      }
      for (auto feature_info : response.default_edge_feature()) {
        std::shared_ptr<Tensor> tensor;
        RETURN_IF_NOT_OK(PbToTensor(&feature_info.feature(), &tensor));
        default_edge_feature_map_[feature_info.type()] = tensor;
      }
    } else {
      RETURN_STATUS_UNEXPECTED(response.error_msg());
    }
  } else {
    auto error_code = status.error_code();
    RETURN_STATUS_UNEXPECTED(status.error_message() + ". GRPC Code " + std::to_string(error_code));
  }
  return Status::OK();
}

Status GraphDataClient::UnRegisterToServer() {
  RETURN_IF_NOT_OK(CheckPid());
  MS_LOG(INFO) << "Graph data client send unregistered to server ";
  void *tag;
  bool ok;
  grpc::Status status;
  grpc::ClientContext ctx;
  grpc::CompletionQueue cq;
  GnnClientUnRegisterRequestPb request;
  GnnClientUnRegisterResponsePb response;
  request.set_pid(static_cast<google::protobuf::int32>(pid_));
  // One minute timeout
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(60);
  ctx.set_deadline(deadline);
  std::unique_ptr<grpc::ClientAsyncResponseReader<GnnClientUnRegisterResponsePb>> rpc(
    stub_->PrepareAsyncClientUnRegister(&ctx, request, &cq));
  rpc->StartCall();
  rpc->Finish(&response, &status, &response);
  {
    py::gil_scoped_release gil_release;
    auto success = cq.Next(&tag, &ok);
    CHECK_FAIL_RETURN_UNEXPECTED(success, "Expect successful");
    CHECK_FAIL_RETURN_UNEXPECTED(tag == &response, "Expect the same tag");
    CHECK_FAIL_RETURN_UNEXPECTED(ok, "Expect successful");
  }
  if (status.ok()) {
    if (response.error_msg() == "Success") {
      MS_LOG(INFO) << "Unregister success.";
      registered_ = false;
    } else {
      RETURN_STATUS_UNEXPECTED(response.error_msg());
    }
  } else {
    auto error_code = status.error_code();
    RETURN_STATUS_UNEXPECTED(status.error_message() + ". GRPC Code " + std::to_string(error_code));
  }
  return Status::OK();
}

Status GraphDataClient::InitFeatureParser() {
  // get shared memory
  graph_shared_memory_ = std::make_unique<GraphSharedMemory>(shared_memory_size_, shared_memory_key_);
  RETURN_IF_NOT_OK(graph_shared_memory_->GetSharedMemory());
  // build feature parser
  graph_feature_parser_ = std::make_unique<GraphFeatureParser>(ShardColumn(data_schema_));

  return Status::OK();
}
#endif

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
