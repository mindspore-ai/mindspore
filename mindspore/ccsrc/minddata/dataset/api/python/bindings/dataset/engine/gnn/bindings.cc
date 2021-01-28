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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/engine/gnn/graph_data_client.h"
#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/engine/gnn/graph_data_server.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  Graph, 0, ([](const py::module *m) {
    (void)py::class_<gnn::GraphData, std::shared_ptr<gnn::GraphData>>(*m, "GraphDataClient")
      .def(py::init([](const std::string &dataset_file, int32_t num_workers, const std::string &working_mode,
                       const std::string &hostname, int32_t port) {
        std::shared_ptr<gnn::GraphData> out;
        if (working_mode == "local") {
          out = std::make_shared<gnn::GraphDataImpl>(dataset_file, num_workers);
        } else if (working_mode == "client") {
          out = std::make_shared<gnn::GraphDataClient>(dataset_file, hostname, port);
        }
        THROW_IF_ERROR(out->Init());
        return out;
      }))
      .def("get_all_nodes",
           [](gnn::GraphData &g, gnn::NodeType node_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllNodes(node_type, &out));
             return out;
           })
      .def("get_all_edges",
           [](gnn::GraphData &g, gnn::EdgeType edge_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllEdges(edge_type, &out));
             return out;
           })
      .def("get_nodes_from_edges",
           [](gnn::GraphData &g, std::vector<gnn::NodeIdType> edge_list) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetNodesFromEdges(edge_list, &out));
             return out;
           })
      .def("get_all_neighbors",
           [](gnn::GraphData &g, std::vector<gnn::NodeIdType> node_list, gnn::NodeType neighbor_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllNeighbors(node_list, neighbor_type, &out));
             return out;
           })
      .def("get_sampled_neighbors",
           [](gnn::GraphData &g, std::vector<gnn::NodeIdType> node_list, std::vector<gnn::NodeIdType> neighbor_nums,
              std::vector<gnn::NodeType> neighbor_types, SamplingStrategy strategy) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetSampledNeighbors(node_list, neighbor_nums, neighbor_types, strategy, &out));
             return out;
           })
      .def("get_neg_sampled_neighbors",
           [](gnn::GraphData &g, std::vector<gnn::NodeIdType> node_list, gnn::NodeIdType neighbor_num,
              gnn::NodeType neg_neighbor_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetNegSampledNeighbors(node_list, neighbor_num, neg_neighbor_type, &out));
             return out;
           })
      .def("get_node_feature",
           [](gnn::GraphData &g, std::shared_ptr<Tensor> node_list, std::vector<gnn::FeatureType> feature_types) {
             TensorRow out;
             THROW_IF_ERROR(g.GetNodeFeature(node_list, feature_types, &out));
             return out.getRow();
           })
      .def("get_edge_feature",
           [](gnn::GraphData &g, std::shared_ptr<Tensor> edge_list, std::vector<gnn::FeatureType> feature_types) {
             TensorRow out;
             THROW_IF_ERROR(g.GetEdgeFeature(edge_list, feature_types, &out));
             return out.getRow();
           })
      .def("graph_info",
           [](gnn::GraphData &g) {
             py::dict out;
             THROW_IF_ERROR(g.GraphInfo(&out));
             return out;
           })
      .def("random_walk",
           [](gnn::GraphData &g, std::vector<gnn::NodeIdType> node_list, std::vector<gnn::NodeType> meta_path,
              float step_home_param, float step_away_param, gnn::NodeIdType default_node) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.RandomWalk(node_list, meta_path, step_home_param, step_away_param, default_node, &out));
             return out;
           })
      .def("stop", [](gnn::GraphData &g) { THROW_IF_ERROR(g.Stop()); });

    (void)py::class_<gnn::GraphDataServer, std::shared_ptr<gnn::GraphDataServer>>(*m, "GraphDataServer")
      .def(py::init([](const std::string &dataset_file, int32_t num_workers, const std::string &hostname, int32_t port,
                       int32_t client_num, bool auto_shutdown) {
        std::shared_ptr<gnn::GraphDataServer> out;
        out =
          std::make_shared<gnn::GraphDataServer>(dataset_file, num_workers, hostname, port, client_num, auto_shutdown);
        THROW_IF_ERROR(out->Init());
        return out;
      }))
      .def("stop", [](gnn::GraphDataServer &g) { THROW_IF_ERROR(g.Stop()); })
      .def("is_stopped", [](gnn::GraphDataServer &g) { return g.IsStopped(); });
  }));

PYBIND_REGISTER(SamplingStrategy, 0, ([](const py::module *m) {
                  (void)py::enum_<SamplingStrategy>(*m, "SamplingStrategy", py::arithmetic())
                    .value("DE_SAMPLING_RANDOM", SamplingStrategy::kRandom)
                    .value("DE_SAMPLING_EDGE_WEIGHT", SamplingStrategy::kEdgeWeight)
                    .export_values();
                }));

}  // namespace dataset
}  // namespace mindspore
