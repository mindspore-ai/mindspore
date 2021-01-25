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
#include "minddata/dataset/engine/gnn/graph_data_service_impl.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/engine/gnn/tensor_proto.h"
#include "minddata/dataset/engine/gnn/graph_data_server.h"

namespace mindspore {
namespace dataset {
namespace gnn {

using pFunction = Status (GraphDataServiceImpl::*)(const GnnGraphDataRequestPb *, GnnGraphDataResponsePb *);
static std::unordered_map<uint32_t, pFunction> g_get_graph_data_func_ = {
  {GET_ALL_NODES, &GraphDataServiceImpl::GetAllNodes},
  {GET_ALL_EDGES, &GraphDataServiceImpl::GetAllEdges},
  {GET_NODES_FROM_EDGES, &GraphDataServiceImpl::GetNodesFromEdges},
  {GET_ALL_NEIGHBORS, &GraphDataServiceImpl::GetAllNeighbors},
  {GET_SAMPLED_NEIGHBORS, &GraphDataServiceImpl::GetSampledNeighbors},
  {GET_NEG_SAMPLED_NEIGHBORS, &GraphDataServiceImpl::GetNegSampledNeighbors},
  {RANDOM_WALK, &GraphDataServiceImpl::RandomWalk},
  {GET_NODE_FEATURE, &GraphDataServiceImpl::GetNodeFeature},
  {GET_EDGE_FEATURE, &GraphDataServiceImpl::GetEdgeFeature}};

GraphDataServiceImpl::GraphDataServiceImpl(GraphDataServer *server, GraphDataImpl *graph_data_impl)
    : server_(server), graph_data_impl_(graph_data_impl) {}

Status GraphDataServiceImpl::FillDefaultFeature(GnnClientRegisterResponsePb *response) {
  const auto default_node_features = graph_data_impl_->GetAllDefaultNodeFeatures();
  for (const auto feature : *default_node_features) {
    GnnFeatureInfoPb *feature_info = response->add_default_node_feature();
    feature_info->set_type(feature.first);
    RETURN_IF_NOT_OK(TensorToPb(feature.second->Value(), feature_info->mutable_feature()));
  }
  const auto default_edge_features = graph_data_impl_->GetAllDefaultEdgeFeatures();
  for (const auto feature : *default_edge_features) {
    GnnFeatureInfoPb *feature_info = response->add_default_edge_feature();
    feature_info->set_type(feature.first);
    RETURN_IF_NOT_OK(TensorToPb(feature.second->Value(), feature_info->mutable_feature()));
  }
  return Status::OK();
}

grpc::Status GraphDataServiceImpl::ClientRegister(grpc::ServerContext *context,
                                                  const GnnClientRegisterRequestPb *request,
                                                  GnnClientRegisterResponsePb *response) {
  Status s = server_->ClientRegister(request->pid());
  if (s.IsOk()) {
    switch (server_->state()) {
      case GraphDataServer::kGdsUninit:
      case GraphDataServer::kGdsInitializing:
        response->set_error_msg("Initializing");
        break;
      case GraphDataServer::kGdsRunning:
        response->set_error_msg("Success");
        response->set_data_schema(graph_data_impl_->GetDataSchema());
        response->set_shared_memory_key(graph_data_impl_->GetSharedMemoryKey());
        response->set_shared_memory_size(graph_data_impl_->GetSharedMemorySize());
        s = FillDefaultFeature(response);
        if (!s.IsOk()) {
          response->set_error_msg(s.ToString());
        }
        break;
      case GraphDataServer::kGdsStopped:
        response->set_error_msg("Stopped");
        break;
    }
  } else {
    response->set_error_msg(s.ToString());
  }
  return ::grpc::Status::OK;
}

grpc::Status GraphDataServiceImpl::ClientUnRegister(grpc::ServerContext *context,
                                                    const GnnClientUnRegisterRequestPb *request,
                                                    GnnClientUnRegisterResponsePb *response) {
  Status s = server_->ClientUnRegister(request->pid());
  if (s.IsOk()) {
    response->set_error_msg("Success");
  } else {
    response->set_error_msg(s.ToString());
  }
  return ::grpc::Status::OK;
}

grpc::Status GraphDataServiceImpl::GetGraphData(grpc::ServerContext *context, const GnnGraphDataRequestPb *request,
                                                GnnGraphDataResponsePb *response) {
  Status s;
  auto iter = g_get_graph_data_func_.find(request->op_name());
  if (iter != g_get_graph_data_func_.end()) {
    pFunction func = iter->second;
    s = (this->*func)(request, response);
    if (s.IsOk()) {
      response->set_error_msg("Success");
    } else {
      response->set_error_msg(s.ToString());
    }
  } else {
    response->set_error_msg("Invalid op name.");
  }
  return ::grpc::Status::OK;
}

grpc::Status GraphDataServiceImpl::GetMetaInfo(grpc::ServerContext *context, const GnnMetaInfoRequestPb *request,
                                               GnnMetaInfoResponsePb *response) {
  MetaInfo meta_info;
  Status s = graph_data_impl_->GetMetaInfo(&meta_info);
  if (s.IsOk()) {
    response->set_error_msg("Success");
    for (const auto &type : meta_info.node_type) {
      auto node_info = response->add_node_info();
      node_info->set_type(static_cast<google::protobuf::int32>(type));
      auto itr = meta_info.node_num.find(type);
      if (itr != meta_info.node_num.end()) {
        node_info->set_num(static_cast<google::protobuf::int32>(itr->second));
      } else {
        node_info->set_num(0);
      }
    }
    for (const auto &type : meta_info.edge_type) {
      auto edge_info = response->add_edge_info();
      edge_info->set_type(static_cast<google::protobuf::int32>(type));
      auto itr = meta_info.edge_num.find(type);
      if (itr != meta_info.edge_num.end()) {
        edge_info->set_num(static_cast<google::protobuf::int32>(itr->second));
      } else {
        edge_info->set_num(0);
      }
    }
    for (const auto &type : meta_info.node_feature_type) {
      response->add_node_feature_type(static_cast<google::protobuf::int32>(type));
    }
    for (const auto &type : meta_info.edge_feature_type) {
      response->add_edge_feature_type(static_cast<google::protobuf::int32>(type));
    }
  } else {
    response->set_error_msg(s.ToString());
  }
  return ::grpc::Status::OK;
}

Status GraphDataServiceImpl::GetAllNodes(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() == 1, "The number of edge types is not 1");

  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetAllNodes(static_cast<NodeType>(request->type()[0]), &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetAllEdges(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() == 1, "The number of edge types is not 1");

  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetAllEdges(static_cast<EdgeType>(request->type()[0]), &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetNodesFromEdges(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->id_size() > 0, "The input edge id is empty");

  std::vector<EdgeIdType> edge_list;
  edge_list.resize(request->id().size());
  std::transform(request->id().begin(), request->id().end(), edge_list.begin(),
                 [](const google::protobuf::int32 id) { return static_cast<EdgeIdType>(id); });
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetNodesFromEdges(edge_list, &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetAllNeighbors(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->id_size() > 0, "The input node id is empty");
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() == 1, "The number of edge types is not 1");

  std::vector<NodeIdType> node_list;
  node_list.resize(request->id().size());
  std::transform(request->id().begin(), request->id().end(), node_list.begin(),
                 [](const google::protobuf::int32 id) { return static_cast<NodeIdType>(id); });
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetAllNeighbors(node_list, static_cast<NodeType>(request->type()[0]), &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetSampledNeighbors(const GnnGraphDataRequestPb *request,
                                                 GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->id_size() > 0, "The input node id is empty");
  CHECK_FAIL_RETURN_UNEXPECTED(request->number_size() > 0, "The input neighbor number is empty");
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() > 0, "The input neighbor type is empty");

  std::vector<NodeIdType> node_list;
  node_list.resize(request->id().size());
  std::transform(request->id().begin(), request->id().end(), node_list.begin(),
                 [](const google::protobuf::int32 id) { return static_cast<NodeIdType>(id); });
  std::vector<NodeIdType> neighbor_nums;
  neighbor_nums.resize(request->number().size());
  std::transform(request->number().begin(), request->number().end(), neighbor_nums.begin(),
                 [](const google::protobuf::int32 num) { return static_cast<NodeIdType>(num); });
  std::vector<NodeType> neighbor_types;
  neighbor_types.resize(request->type().size());
  std::transform(request->type().begin(), request->type().end(), neighbor_types.begin(),
                 [](const google::protobuf::int32 type) { return static_cast<NodeType>(type); });
  SamplingStrategy strategy = static_cast<SamplingStrategy>(request->strategy());
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetSampledNeighbors(node_list, neighbor_nums, neighbor_types, strategy, &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetNegSampledNeighbors(const GnnGraphDataRequestPb *request,
                                                    GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->id_size() > 0, "The input node id is empty");
  CHECK_FAIL_RETURN_UNEXPECTED(request->number_size() == 1, "The number of neighbor number is not 1");
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() == 1, "The number of neighbor types is not 1");

  std::vector<NodeIdType> node_list;
  node_list.resize(request->id().size());
  std::transform(request->id().begin(), request->id().end(), node_list.begin(),
                 [](const google::protobuf::int32 id) { return static_cast<NodeIdType>(id); });
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->GetNegSampledNeighbors(node_list, static_cast<NodeIdType>(request->number()[0]),
                                                            static_cast<NodeType>(request->type()[0]), &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::RandomWalk(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  CHECK_FAIL_RETURN_UNEXPECTED(request->id_size() > 0, "The input node id is empty");
  CHECK_FAIL_RETURN_UNEXPECTED(request->type_size() > 0, "The input meta path is empty");

  std::vector<NodeIdType> node_list;
  node_list.resize(request->id().size());
  std::transform(request->id().begin(), request->id().end(), node_list.begin(),
                 [](const google::protobuf::int32 id) { return static_cast<NodeIdType>(id); });
  std::vector<NodeType> meta_path;
  meta_path.resize(request->type().size());
  std::transform(request->type().begin(), request->type().end(), meta_path.begin(),
                 [](const google::protobuf::int32 type) { return static_cast<NodeType>(type); });
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(graph_data_impl_->RandomWalk(node_list, meta_path, request->random_walk().p(),
                                                request->random_walk().q(), request->random_walk().default_id(),
                                                &tensor));
  TensorPb *result = response->add_result_data();
  RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  return Status::OK();
}

Status GraphDataServiceImpl::GetNodeFeature(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  std::shared_ptr<Tensor> nodes;
  RETURN_IF_NOT_OK(PbToTensor(&request->id_tensor(), &nodes));
  for (const auto &type : request->type()) {
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(graph_data_impl_->GetNodeFeatureSharedMemory(nodes, type, &tensor));
    TensorPb *result = response->add_result_data();
    RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  }
  return Status::OK();
}

Status GraphDataServiceImpl::GetEdgeFeature(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response) {
  std::shared_ptr<Tensor> edges;
  RETURN_IF_NOT_OK(PbToTensor(&request->id_tensor(), &edges));
  for (const auto &type : request->type()) {
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(graph_data_impl_->GetEdgeFeatureSharedMemory(edges, type, &tensor));
    TensorPb *result = response->add_result_data();
    RETURN_IF_NOT_OK(TensorToPb(tensor, result));
  }
  return Status::OK();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
