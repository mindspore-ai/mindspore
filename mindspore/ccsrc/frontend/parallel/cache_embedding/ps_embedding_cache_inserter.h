/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_EMBEDDING_CACHE_PS_EMBEDDING_CACHE_INSERTER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_EMBEDDING_CACHE_PS_EMBEDDING_CACHE_INSERTER_H_

#include <string>
#include <map>
#include <vector>

#include "ir/anf.h"
#include "include/backend/distributed/constants.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace parallel {
// Build service-side graph for embedding distributed cache based on Parameter Server,
// and remove all nodes of origin func graph.
class PsEmbeddingCacheInserter {
 public:
  PsEmbeddingCacheInserter(const FuncGraphPtr &root_graph, int64_t rank_id, const std::string &node_role,
                           uint32_t worker_num)
      : root_graph_(root_graph), rank_id_(rank_id), node_role_(node_role), worker_num_(worker_num) {}

  ~PsEmbeddingCacheInserter() {
    root_graph_ = nullptr;
    keys_to_params_.clear();
    shapes_to_nodes_.clear();
  }

  // Insert embedding cache sub graphs to replace all nodes of origin func graph.
  bool Run();

 private:
  // Construct the embedding cache graph of server:
  // Recv --> SwitchLayer --> Call --> Return
  // the SwitchLayer is used to select the subgraph corresponding to the service requested to be executed.
  bool ConstructEmbeddingCacheGraph() const;

  // Create RpcRecv node for server to receive request.
  CNodePtr CreateRecvNode() const;

  // Build Embedding store for each param which enable cache. Embedding store can read/write embedding from/to
  // persistent storage.
  void BuildEmbeddingStorages();

  // Construct the embedding cache services subgraphs, including embedding lookup and update operations, and package the
  // subgraphs corresponding to the related operations into the partial.
  bool ConstructEmbeddingCacheServicesSubGraphs(const std::vector<CNodePtr> &recv_outputs,
                                                std::vector<AnfNodePtr> *make_tuple_inputs) const;

  // Construct embedding lookup service sub graph:
  // Input(param, indices) --> EmbeddingLookup/MapTensorGet --> RpcSend --> Return
  // RpcSend is used to send the embeddings to the service caller.
  FuncGraphPtr ConstructEmbeddingLookupSubGraph(const AnfNodePtr &node, const ParameterPtr &param,
                                                int32_t param_key) const;

  // Construct updating embedding service sub graph:
  // Input(param, indices, update_values) --> ScatterUpdate/MapTensorPut --> Return
  // The Sub is used to rectify the id via offset for embedding slice.
  FuncGraphPtr ConstructUpdateEmbeddingSubGraph(const ParameterPtr &param, const AnfNodePtr &node,
                                                int32_t param_key) const;

  // Create embedding lookup kernel: 'EmbeddingLookup' for Tensor or 'MapTensorGet' for Hash Table.
  CNodePtr CreateEmbeddingLookupKernel(const FuncGraphPtr &graph, const ParameterPtr &input_param,
                                       const ParameterPtr &input_indices,
                                       const AnfNodePtr &origin_embedding_lookup_node) const;

  // Create embedding update kernel: 'ScatterUpdate' for Tensor or 'MapTensorPut' for Hash Table.
  CNodePtr CreateEmbeddingUpdateKernel(const FuncGraphPtr &graph, const ParameterPtr &input_param,
                                       const ParameterPtr &input_indices, const ParameterPtr &update_values) const;

  // Create return node for subgraph, using depend node to return a fake value node to ensure that the output abstract
  // of each subgraph is the same.
  CNodePtr CreateReturnNode(const FuncGraphPtr graph, const AnfNodePtr &output_node) const;

  // Set attr(device target attr and graph split label) for all CNodes.
  void SetAttrForAllNodes() const;

  // Set device target attr to cpu, set graph split label(rank id and node role, such as (0, "MS_PSERVER")).
  void SetNodeAttr(const CNodePtr &node, const std::string &node_role = distributed::kEnvRoleOfPServer) const;

  // Set attrs for send node, such as:inter process edges, send dst ranks, send dst roles.
  void SetSendNodeAttr(const CNodePtr &send_node, int32_t param_key, const std::string &embedding_cache_op,
                       const std::string &dst_role = distributed::kEnvRoleOfWorker) const;

  // Set attrs for recv node, such as:inter process edges, recv src ranks, recv src roles.
  void SetRecvNodeAttr(const CNodePtr &recv_node, const std::string &src_role = distributed::kEnvRoleOfWorker) const;

  // Get EmbeddingLookup nodes which are executed on server from origin function graph.
  void GetEmbeddingLookupNodes();

  // Get parameters enabled embedding cache of origin function graph.
  void GetCacheEnableParameters();

  // Origin root function graph.
  FuncGraphPtr root_graph_;

  // The rank id of this process.
  int64_t rank_id_;
  // The node role of this process.
  std::string node_role_;
  // The worker number of in cluster.
  uint32_t worker_num_;

  // Record parameters enabled embedding cache of origin function graph.
  // Key: parameter key, Value: ParameterPtr
  std::map<int32_t, ParameterPtr> keys_to_params_;

  // Record EmbeddingLookup nodes which are executed on server from origin function graph.
  // Key: shape of EmbeddingLookup node, Value: EmbeddingLookup AnfNodePtr.
  std::map<ShapeVector, AnfNodePtr> shapes_to_nodes_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_EMBEDDING_CACHE_PS_EMBEDDING_CACHE_INSERTER_H_
