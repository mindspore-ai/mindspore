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

#include "frontend/parallel/cache_embedding/ps_embedding_cache_inserter.h"

#include <memory>
#include <string>
#include <algorithm>

#include "ir/func_graph.h"
#include "abstract/abstract_function.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
// One dimensional shape placeholder.
const ShapeVector kOneDimDynamicShape = {-1};
// Two dimensional shape placeholder.
const ShapeVector kTwoDimsDynamicShape = {-1, -1};
// The output tensor number of recv node.
const size_t kRecvNodeOutputNum = 3;

void PsEmbeddingCacheInserter::GetEmbeddingLookupNodes() {
  MS_EXCEPTION_IF_NULL(root_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(root_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!(node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kEmbeddingLookupOpName)) {
      return;
    }
    const PrimitivePtr &prim = common::AnfAlgo::GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    if (!(prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole))) {
      return;
    }

    int64_t rank_id_attr = GetValue<int64_t>(prim->GetAttr(distributed::kOpLabelRankId));
    std::string node_role_attr = GetValue<std::string>(prim->GetAttr(distributed::kOpLabelRole));
    if (rank_id_attr == rank_id_ && node_role_attr == node_role_) {
      std::vector<size_t> shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
      shapes_to_nodes_[shape] = node;
    }
  });
}

void PsEmbeddingCacheInserter::SetNodeAttr(const CNodePtr &node, const std::string &node_role) const {
  MS_EXCEPTION_IF_NULL(node);

  // Set attr for call node, call node hasn't primitive to save attrs, so save attrs into CNode.
  if (common::AnfAlgo::IsCallNode(node)) {
    node->AddAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice));
    node->AddAttr(distributed::kOpLabelRankId, MakeValue(rank_id_));
    node->AddAttr(distributed::kOpLabelRole, MakeValue(node_role));
  } else {
    common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), node);
    common::AnfAlgo::SetNodeAttr(distributed::kOpLabelRankId, MakeValue(rank_id_), node);
    common::AnfAlgo::SetNodeAttr(distributed::kOpLabelRole, MakeValue(node_role), node);
  }
}

void PsEmbeddingCacheInserter::SetSendNodeAttr(const CNodePtr &send_node, int32_t param_key,
                                               const std::string &embedding_cache_op,
                                               const std::string &dst_role) const {
  MS_EXCEPTION_IF_NULL(send_node);
  SetNodeAttr(send_node);

  std::vector<uint32_t> dst_ranks;
  std::vector<std::string> dst_roles = {dst_role};
  std::vector<std::string> inter_process_edges;

  // Set inter process edges, send dst ranks, send dst roles.
  for (uint32_t i = 0; i < worker_num_; i++) {
    dst_ranks.push_back(i);
    dst_roles.push_back(dst_role);
    // Unique edge name: src role + src rank id -> dst role + dst rank id +embedding cache operation + parameter key.
    inter_process_edges.push_back(distributed::kEnvRoleOfServer + std::to_string(rank_id_) + "->" + dst_role +
                                  std::to_string(i) + "_" + embedding_cache_op + "_" + distributed::kParameterKey +
                                  std::to_string(param_key));
  }

  common::AnfAlgo::SetNodeAttr(kAttrSendDstRanks, MakeValue(dst_ranks), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRoles, MakeValue(dst_roles), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), send_node);
}

void PsEmbeddingCacheInserter::SetRecvNodeAttr(const CNodePtr &recv_node, const std::string &src_role) const {
  MS_EXCEPTION_IF_NULL(recv_node);
  SetNodeAttr(recv_node);

  std::vector<uint32_t> src_ranks;
  std::vector<std::string> src_roles;
  std::vector<std::string> inter_process_edges;

  // Set inter process edges, recv src ranks, recv src roles.
  // Each server has only one Recv node, which needs to receive all requests from each worker. For example, different
  // parameters on each worker have two operations: look up embedding and update embedding. Each operation will be
  // performed by an independent Send node, so the Recv node on the server side will have multiple edges.
  for (uint32_t i = 0; i < worker_num_; i++) {
    for (const auto &item : keys_to_params_) {
      int32_t param_key = item.first;
      for (uint32_t k = 0; k < distributed::kEmbeddingCacheOps.size(); k++) {
        src_ranks.push_back(i);
        src_roles.push_back(src_role);
        // Unique edge name: src role + src rank id -> dst role + dst rank id + embedding cache operation + parameter
        // key.
        inter_process_edges.push_back(src_role + std::to_string(i) + "->" + distributed::kEnvRoleOfServer +
                                      std::to_string(rank_id_) + "_" + distributed::kEmbeddingCacheOps[k] + "_" +
                                      distributed::kParameterKey + std::to_string(param_key));
      }
    }
  }

  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRanks, MakeValue(src_ranks), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRoles, MakeValue(src_roles), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), recv_node);
}

CNodePtr PsEmbeddingCacheInserter::CreateReturnNode(const FuncGraphPtr graph, const AnfNodePtr &output_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(output_node);

  // Create fake output value node to make sure the output abstract is the same for each subgraph.
  auto fake_output_tensor = std::make_shared<tensor::Tensor>(1.0);
  auto fake_output_value = NewValueNode(fake_output_tensor);
  MS_EXCEPTION_IF_NULL(fake_output_value);
  fake_output_value->set_abstract(fake_output_tensor->ToAbstract());

  // Create depend node.
  auto depend_node = graph->NewCNode({NewValueNode(prim::kPrimDepend), fake_output_value, output_node});
  MS_EXCEPTION_IF_NULL(depend_node);

  // Create return node.
  std::vector<AnfNodePtr> return_inputs;
  return_inputs.push_back(NewValueNode(prim::kPrimReturn));
  return_inputs.push_back(depend_node);
  auto return_node = graph->NewCNode(return_inputs);
  MS_EXCEPTION_IF_NULL(return_node);

  return return_node;
}

FuncGraphPtr PsEmbeddingCacheInserter::ConstructEmbeddingLookupSubGraph(const AnfNodePtr &node,
                                                                        const ParameterPtr &param,
                                                                        int32_t param_key) const {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(node);

  // 1. Create subgraph and parameters.
  auto graph = std::make_shared<FuncGraph>();
  ParameterPtr input_param = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(input_param);
  MS_EXCEPTION_IF_NULL(param->abstract());
  input_param->set_abstract(param->abstract()->Clone());
  ParameterPtr input_indices = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(input_indices);
  input_indices->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt32, kOneDimDynamicShape));

  // 2. Create EmbeddingLookup node.
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kEmbeddingLookupOpName);
  std::vector<AnfNodePtr> emb_lookup_inputs{NewValueNode(emb_lookup_primitive), input_param, input_indices};
  auto embedding_cache_lookup_node = graph->NewCNode(emb_lookup_inputs);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node);
  common::AnfAlgo::CopyNodeAttrs(node, embedding_cache_lookup_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), embedding_cache_lookup_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), embedding_cache_lookup_node);
  SetNodeAttr(embedding_cache_lookup_node);

  // 3. Create RpcSend node.
  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  send_inputs.push_back(embedding_cache_lookup_node);
  CNodePtr send_node = graph->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(send_node);
  SetSendNodeAttr(send_node, param_key, distributed::kLookupEmbeddingCache);

  // 4. Create return node.
  CNodePtr return_node = CreateReturnNode(graph, send_node);
  MS_EXCEPTION_IF_NULL(return_node);
  graph->set_return(return_node);

  MS_EXCEPTION_IF_NULL(root_graph_);
  auto manager = root_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(graph);
  return graph;
}

FuncGraphPtr PsEmbeddingCacheInserter::ConstructUpdateEmbeddingSubGraph(const ParameterPtr &param,
                                                                        const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(node);

  // 1. Create subgraph and parameters.
  auto graph = std::make_shared<FuncGraph>();
  ParameterPtr input_param = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(input_param);
  MS_EXCEPTION_IF_NULL(param->abstract());
  input_param->set_abstract(param->abstract()->Clone());
  ParameterPtr input_indices = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(input_indices);
  input_indices->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt32, kOneDimDynamicShape));
  ParameterPtr update_values = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(update_values);
  update_values->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimsDynamicShape));

  // 2. Create Sub node.
  int32_t offset = LongToInt(common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrOffset));
  PrimitivePtr sub_primitive = std::make_shared<Primitive>(kSubOpName);
  std::vector<AnfNodePtr> sub_inputs{NewValueNode(sub_primitive), input_indices, NewValueNode(MakeValue(offset))};
  auto sub_node = graph->NewCNode(sub_inputs);
  MS_EXCEPTION_IF_NULL(sub_node);
  SetNodeAttr(sub_node);

  // 3. Create ScatterUpdate node.
  PrimitivePtr embedding_cache_update_primitive = std::make_shared<Primitive>(kScatterUpdateOpName);
  std::vector<AnfNodePtr> embedding_cache_update_inputs{NewValueNode(embedding_cache_update_primitive), input_param,
                                                        sub_node, update_values};
  auto embedding_cache_update_node = graph->NewCNode(embedding_cache_update_inputs);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node);
  common::AnfAlgo::CopyNodeAttrs(node, embedding_cache_update_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), embedding_cache_update_node);
  SetNodeAttr(embedding_cache_update_node);

  // 4. Create return node.
  CNodePtr return_node = CreateReturnNode(graph, embedding_cache_update_node);
  MS_EXCEPTION_IF_NULL(return_node);
  graph->set_return(return_node);

  MS_EXCEPTION_IF_NULL(root_graph_);
  auto manager = root_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(graph);
  return graph;
}

CNodePtr PsEmbeddingCacheInserter::CreateRecvNode() const {
  // 1. Create abstract for RpcRecv node.
  auto indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, kOneDimDynamicShape);
  auto update_values_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimsDynamicShape);
  auto fake_id_tensor = std::make_shared<tensor::Tensor>(static_cast<int64_t>(0));

  // 2. Create fake input nodes for RpcRecv node.
  // The indices input.
  MS_EXCEPTION_IF_NULL(indices_abstract->element());
  MS_EXCEPTION_IF_NULL(indices_abstract->element()->BuildType());
  MS_EXCEPTION_IF_NULL(indices_abstract->shape());
  auto fake_indices_tensor = std::make_shared<tensor::Tensor>(indices_abstract->element()->BuildType()->type_id(),
                                                              indices_abstract->shape()->shape());
  auto fake_indices_value = NewValueNode(fake_indices_tensor);
  MS_EXCEPTION_IF_NULL(fake_indices_value);
  fake_indices_value->set_abstract(fake_indices_tensor->ToAbstract());

  // The update values input.
  MS_EXCEPTION_IF_NULL(update_values_abstract->element());
  MS_EXCEPTION_IF_NULL(update_values_abstract->element()->BuildType());
  MS_EXCEPTION_IF_NULL(update_values_abstract->shape());
  auto fake_update_values_tensor = std::make_shared<tensor::Tensor>(
    update_values_abstract->element()->BuildType()->type_id(), update_values_abstract->shape()->shape());
  auto fake_update_values_value = NewValueNode(fake_update_values_tensor);
  MS_EXCEPTION_IF_NULL(fake_update_values_value);
  fake_update_values_value->set_abstract(fake_update_values_tensor->ToAbstract());

  // The id input, id is used to choose service.
  auto fake_id_value = NewValueNode(fake_id_tensor);
  MS_EXCEPTION_IF_NULL(fake_id_value);
  fake_id_value->set_abstract(fake_id_tensor->ToAbstract());

  // 3. Create a RpcRecv node.
  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  recv_inputs.push_back(fake_indices_value);
  recv_inputs.push_back(fake_update_values_value);
  recv_inputs.push_back(fake_id_value);
  MS_EXCEPTION_IF_NULL(root_graph_);
  CNodePtr recv_node = root_graph_->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);
  SetRecvNodeAttr(recv_node);

  return recv_node;
}

bool PsEmbeddingCacheInserter::ConstructEmbeddingCacheServicesSubGraphs(
  const std::vector<CNodePtr> &recv_outputs, std::vector<AnfNodePtr> *make_tuple_inputs) const {
  MS_EXCEPTION_IF_NULL(root_graph_);
  MS_EXCEPTION_IF_NULL(make_tuple_inputs);
  if (recv_outputs.size() != kRecvNodeOutputNum) {
    MS_LOG(ERROR) << "The output tensor number of recv node is not equal to " << kRecvNodeOutputNum;
    return false;
  }

  for (const auto &item : keys_to_params_) {
    int32_t key = item.first;
    ParameterPtr param = item.second;
    MS_EXCEPTION_IF_NULL(param);
    auto shape = common::AnfAlgo::GetOutputInferShape(param, 0);
    auto iter = shapes_to_nodes_.find(shape);
    if (iter == shapes_to_nodes_.end()) {
      MS_LOG(ERROR) << "Can not find cnode for parameter(key[" << key << "]) with shape: " << shape;
      return false;
    }
    AnfNodePtr node = iter->second;

    // 1. Construct embedding lookup service sub graph.
    auto emb_lookup_sub_graph = ConstructEmbeddingLookupSubGraph(node, param, key);
    MS_EXCEPTION_IF_NULL(emb_lookup_sub_graph);
    auto emb_lookup_graph_value = NewValueNode(emb_lookup_sub_graph);
    MS_EXCEPTION_IF_NULL(emb_lookup_graph_value);
    auto emb_lookup_graph_value_abstract = std::make_shared<abstract::FuncGraphAbstractClosure>(
      emb_lookup_sub_graph, abstract::AnalysisContext::DummyContext());
    emb_lookup_graph_value->set_abstract(emb_lookup_graph_value_abstract);

    CNodePtr emb_lookup_partial_node =
      root_graph_->NewCNode({NewValueNode(prim::kPrimPartial), emb_lookup_graph_value, param, recv_outputs[0]});
    MS_EXCEPTION_IF_NULL(emb_lookup_partial_node);
    make_tuple_inputs->push_back(emb_lookup_partial_node);

    // 2. Construct updating embedding service sub graph.
    auto update_emb_sub_graph = ConstructUpdateEmbeddingSubGraph(param, node);
    MS_EXCEPTION_IF_NULL(update_emb_sub_graph);
    auto update_emb_graph_value = NewValueNode(update_emb_sub_graph);
    MS_EXCEPTION_IF_NULL(update_emb_graph_value);
    auto update_emb_graph_value_abstract = std::make_shared<abstract::FuncGraphAbstractClosure>(
      update_emb_sub_graph, abstract::AnalysisContext::DummyContext());
    update_emb_graph_value->set_abstract(update_emb_graph_value_abstract);

    CNodePtr update_emb_partial_node = root_graph_->NewCNode(
      {NewValueNode(prim::kPrimPartial), update_emb_graph_value, param, recv_outputs[0], recv_outputs[1]});
    MS_EXCEPTION_IF_NULL(update_emb_partial_node);
    make_tuple_inputs->push_back(update_emb_partial_node);
  }

  return true;
}

bool PsEmbeddingCacheInserter::ConstructEmbeddingCacheGraph() const {
  // 1. Create recv node for server.
  CNodePtr recv_node = CreateRecvNode();
  MS_EXCEPTION_IF_NULL(recv_node);
  auto value_node_0 = NewValueNode(static_cast<int64_t>(0));
  auto value_node_1 = NewValueNode(static_cast<int64_t>(1));
  auto value_node_2 = NewValueNode(static_cast<int64_t>(2));
  std::vector<AnfNodePtr> getitem_input0{NewValueNode(prim::kPrimTupleGetItem), recv_node, value_node_0};
  std::vector<AnfNodePtr> getitem_input1{NewValueNode(prim::kPrimTupleGetItem), recv_node, value_node_1};
  std::vector<AnfNodePtr> getitem_input2{NewValueNode(prim::kPrimTupleGetItem), recv_node, value_node_2};

  MS_EXCEPTION_IF_NULL(root_graph_);
  auto getitem_0 = root_graph_->NewCNode(getitem_input0);
  auto getitem_1 = root_graph_->NewCNode(getitem_input1);
  auto getitem_2 = root_graph_->NewCNode(getitem_input2);
  // The tuple_getitem nodes used to get the outputs of recv node.
  std::vector<CNodePtr> getitems = {getitem_0, getitem_1, getitem_2};

  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};

  // 2. Construct the embedding cache services subgraphs, including embedding lookup and update operations, and
  // package the subgraphs corresponding to the related operations into the partial.
  RETURN_IF_FALSE_WITH_LOG(ConstructEmbeddingCacheServicesSubGraphs(getitems, &make_tuple_inputs),
                           "Construct embedding cache services sub graphs failed.");

  auto make_tuple_node = root_graph_->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  // 3. Create switch layer and call node, used to select and execute the subgraph corresponding to the service
  // requested.
  std::vector<AnfNodePtr> switch_layer_inputs = {NewValueNode(prim::kPrimSwitchLayer), getitem_2, make_tuple_node};
  auto switch_layer_node = root_graph_->NewCNode(switch_layer_inputs);

  CNodePtr call_node = root_graph_->NewCNode({switch_layer_node});
  MS_EXCEPTION_IF_NULL(call_node);

  // 4. Replace useless nodes of origin function graph.
  auto graph_manager = root_graph_->manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  graph_manager->Replace(root_graph_->output(), call_node);
  auto return_node = root_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  return true;
}

bool PsEmbeddingCacheInserter::Run() {
  // Get EmbeddingLookup nodes which are executed on server from origin function graph.
  GetEmbeddingLookupNodes();

  // Construct the embedding cache graph of server.
  RETURN_IF_FALSE_WITH_LOG(ConstructEmbeddingCacheGraph(), "Construct embedding cache graph failed.");
  return true;
}
}  // namespace parallel
}  // namespace mindspore
