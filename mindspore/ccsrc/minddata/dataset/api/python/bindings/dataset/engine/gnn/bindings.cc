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

#include "minddata/dataset/engine/gnn/graph.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  Graph, 0, ([](const py::module *m) {
    (void)py::class_<gnn::Graph, std::shared_ptr<gnn::Graph>>(*m, "Graph")
      .def(py::init([](std::string dataset_file, int32_t num_workers) {
        std::shared_ptr<gnn::Graph> g_out = std::make_shared<gnn::Graph>(dataset_file, num_workers);
        THROW_IF_ERROR(g_out->Init());
        return g_out;
      }))
      .def("get_all_nodes",
           [](gnn::Graph &g, gnn::NodeType node_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllNodes(node_type, &out));
             return out;
           })
      .def("get_all_edges",
           [](gnn::Graph &g, gnn::EdgeType edge_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllEdges(edge_type, &out));
             return out;
           })
      .def("get_nodes_from_edges",
           [](gnn::Graph &g, std::vector<gnn::NodeIdType> edge_list) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetNodesFromEdges(edge_list, &out));
             return out;
           })
      .def("get_all_neighbors",
           [](gnn::Graph &g, std::vector<gnn::NodeIdType> node_list, gnn::NodeType neighbor_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetAllNeighbors(node_list, neighbor_type, &out));
             return out;
           })
      .def("get_sampled_neighbors",
           [](gnn::Graph &g, std::vector<gnn::NodeIdType> node_list, std::vector<gnn::NodeIdType> neighbor_nums,
              std::vector<gnn::NodeType> neighbor_types) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetSampledNeighbors(node_list, neighbor_nums, neighbor_types, &out));
             return out;
           })
      .def("get_neg_sampled_neighbors",
           [](gnn::Graph &g, std::vector<gnn::NodeIdType> node_list, gnn::NodeIdType neighbor_num,
              gnn::NodeType neg_neighbor_type) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.GetNegSampledNeighbors(node_list, neighbor_num, neg_neighbor_type, &out));
             return out;
           })
      .def("get_node_feature",
           [](gnn::Graph &g, std::shared_ptr<Tensor> node_list, std::vector<gnn::FeatureType> feature_types) {
             TensorRow out;
             THROW_IF_ERROR(g.GetNodeFeature(node_list, feature_types, &out));
             return out.getRow();
           })
      .def("get_edge_feature",
           [](gnn::Graph &g, std::shared_ptr<Tensor> edge_list, std::vector<gnn::FeatureType> feature_types) {
             TensorRow out;
             THROW_IF_ERROR(g.GetEdgeFeature(edge_list, feature_types, &out));
             return out.getRow();
           })
      .def("graph_info",
           [](gnn::Graph &g) {
             py::dict out;
             THROW_IF_ERROR(g.GraphInfo(&out));
             return out;
           })
      .def("random_walk",
           [](gnn::Graph &g, std::vector<gnn::NodeIdType> node_list, std::vector<gnn::NodeType> meta_path,
              float step_home_param, float step_away_param, gnn::NodeIdType default_node) {
             std::shared_ptr<Tensor> out;
             THROW_IF_ERROR(g.RandomWalk(node_list, meta_path, step_home_param, step_away_param, default_node, &out));
             return out;
           });
  }));

}  // namespace dataset
}  // namespace mindspore
