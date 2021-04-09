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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVICE_IMPL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVICE_IMPL_H_

#include <memory>
#include <string>

#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "proto/gnn_graph_data.grpc.pb.h"
#include "proto/gnn_graph_data.pb.h"

namespace mindspore {
namespace dataset {
namespace gnn {

class GraphDataServer;

// class GraphDataServiceImpl : public GnnGraphData::Service {
class GraphDataServiceImpl {
 public:
  GraphDataServiceImpl(GraphDataServer *server, GraphDataImpl *graph_data_impl);
  ~GraphDataServiceImpl() = default;

  grpc::Status ClientRegister(grpc::ServerContext *context, const GnnClientRegisterRequestPb *request,
                              GnnClientRegisterResponsePb *response);

  grpc::Status ClientUnRegister(grpc::ServerContext *context, const GnnClientUnRegisterRequestPb *request,
                                GnnClientUnRegisterResponsePb *response);

  grpc::Status GetGraphData(grpc::ServerContext *context, const GnnGraphDataRequestPb *request,
                            GnnGraphDataResponsePb *response);

  grpc::Status GetMetaInfo(grpc::ServerContext *context, const GnnMetaInfoRequestPb *request,
                           GnnMetaInfoResponsePb *response);

  Status GetAllNodes(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetAllEdges(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetNodesFromEdges(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetEdgesFromNodes(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetAllNeighbors(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetSampledNeighbors(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetNegSampledNeighbors(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status RandomWalk(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetNodeFeature(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);
  Status GetEdgeFeature(const GnnGraphDataRequestPb *request, GnnGraphDataResponsePb *response);

 private:
  Status FillDefaultFeature(GnnClientRegisterResponsePb *response);

  GraphDataServer *server_;
  GraphDataImpl *graph_data_impl_;
};

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_SERVICE_IMPL_H_
