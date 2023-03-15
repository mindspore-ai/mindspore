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
#include <utility>

#include "ir/func_graph.h"
#include "abstract/abstract_function.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace parallel {
// One dimensional shape placeholder.
const ShapeVector kOneDimShape = {1};
// Two dimensional shape placeholder.
const ShapeVector kTwoDimsShape = {1, 1};

// One dimensional shape placeholder.
const ShapeVector kOneDimDynamicShape = {-1};
// Two dimensional shape placeholder.
const ShapeVector kTwoDimsDynamicShape = {-1, -1};

// The output tensor number of recv node.
const size_t kRecvNodeOutputNum = 3;

// The input index of offset of EmbeddingLookup kernel.
constexpr size_t kEmbeddingLookupOffsetIdx = 2;

// The dims of embedding table.
constexpr size_t kEmbeddingTableDims = 2;

constexpr char kEmbeddingRemoteCacheNode[] = "EmbeddingRemoteCacheNode";
constexpr char kEmbeddingLocalCacheNode[] = "EmbeddingLocalCacheNode";

namespace {
ValueNodePtr CreateFakeValueNode(const AnfNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  abstract::AbstractTensorPtr origin_abstract = origin_node->abstract()->cast<abstract::AbstractTensorPtr>();

  MS_EXCEPTION_IF_NULL(origin_abstract);
  tensor::TensorPtr fake_tensor = std::make_shared<tensor::Tensor>(origin_abstract->element()->BuildType()->type_id(),
                                                                   origin_abstract->shape()->shape());
  MS_EXCEPTION_IF_NULL(fake_tensor);
  fake_tensor->set_base_shape(origin_abstract->shape()->Clone());

  auto fake_value = NewValueNode(fake_tensor);
  MS_EXCEPTION_IF_NULL(fake_value);
  fake_value->set_abstract(fake_tensor->ToAbstract());
  return fake_value;
}

AnfNodePtr CreateOutputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &origin_output) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_output);
  MS_EXCEPTION_IF_NULL(origin_output->abstract());
  if (origin_output->abstract()->isa<abstract::AbstractTuple>()) {
    abstract::AbstractBasePtrList new_elements_abs;
    std::vector<ValuePtr> new_elements_values;

    auto tuple_elements = origin_output->abstract()->cast<abstract::AbstractTuplePtr>()->elements();
    for (const auto &element : tuple_elements) {
      MS_EXCEPTION_IF_NULL(element);
      auto tensor_abstract = element->cast<abstract::AbstractTensorPtr>();
      if (!tensor_abstract) {
        MS_LOG(EXCEPTION) << "Only support to replace tuple with all tensor elements.";
      }
      auto fake_tensor = std::make_shared<tensor::Tensor>(tensor_abstract->element()->BuildType()->type_id(),
                                                          tensor_abstract->shape()->shape());
      MS_EXCEPTION_IF_NULL(fake_tensor);
      new_elements_abs.push_back(fake_tensor->ToAbstract());
      new_elements_values.push_back(fake_tensor);
    }
    ValueTuplePtr value_tuple = std::make_shared<ValueTuple>(new_elements_values);
    auto value_tuple_abs = std::make_shared<abstract::AbstractTuple>(new_elements_abs);
    auto value_tuple_node = NewValueNode(value_tuple);
    MS_EXCEPTION_IF_NULL(value_tuple_node);
    value_tuple_node->set_abstract(value_tuple_abs);
    return value_tuple_node;
  } else {
    return CreateFakeValueNode(origin_output);
  }
}
}  // namespace

void PsEmbeddingCacheInserter::GetEmbeddingLookupNodes() {
  MS_EXCEPTION_IF_NULL(root_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(root_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      return;
    }

    const std::string kernel_name = common::AnfAlgo::GetCNodeName(node);
    if (kernel_name != kGatherOpName && kernel_name != kSparseGatherV2OpName) {
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
      auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
      shapes_to_nodes_[shape] = node;
    }
  });
}

void PsEmbeddingCacheInserter::GetCacheEnableParameters() {
  MS_EXCEPTION_IF_NULL(root_graph_);
  const std::vector<AnfNodePtr> &parameters = root_graph_->parameters();
  auto params_size = parameters.size();
  for (size_t i = 0; i < params_size; ++i) {
    MS_EXCEPTION_IF_NULL(parameters[i]);
    if (!parameters[i]->isa<Parameter>()) {
      MS_LOG(EXCEPTION) << "The node with name: " << parameters[i]->fullname_with_scope() << "is not a Parameter.";
    }

    ParameterPtr param = parameters[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    auto param_info = param->param_info();
    if (param_info && param_info->key() != -1 && param_info->cache_enable()) {
      keys_to_params_[param_info->key()] = param;
      MS_LOG(INFO) << "Parameter[" << param->fullname_with_scope() << "], key[" << param_info->key() << "]";
    }
  }
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

void PsEmbeddingCacheInserter::SetAttrForAllNodes() const {
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(root_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      return;
    }
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    SetNodeAttr(cnode);
  });
}

void PsEmbeddingCacheInserter::SetSendNodeAttr(const CNodePtr &send_node, int32_t param_key,
                                               const std::string &embedding_cache_op,
                                               const std::string &dst_role) const {
  MS_EXCEPTION_IF_NULL(send_node);

  std::vector<uint32_t> dst_ranks;
  std::vector<std::string> dst_roles = {dst_role};
  std::vector<std::string> inter_process_edges;

  // Set inter process edges, send dst ranks, send dst roles.
  for (uint32_t i = 0; i < worker_num_; i++) {
    dst_ranks.push_back(i);
    dst_roles.push_back(dst_role);
    // Unique edge name: src role + src rank id -> dst role + dst rank id +embedding cache operation + parameter key.
    inter_process_edges.push_back(distributed::kEnvRoleOfPServer + std::to_string(rank_id_) + "->" + dst_role +
                                  std::to_string(i) + "_" + embedding_cache_op + "_" + distributed::kParameterKey +
                                  std::to_string(param_key));
  }

  common::AnfAlgo::SetNodeAttr(kAttrSendDstRanks, MakeValue(dst_ranks), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRoles, MakeValue(dst_roles), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendSrcNodeName, MakeValue(std::string(kEmbeddingRemoteCacheNode)), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendDstNodeName, MakeValue(std::string(kEmbeddingLocalCacheNode)), send_node);

  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrIsMuxRpcKernel, MakeValue(true), send_node);
}

void PsEmbeddingCacheInserter::SetRecvNodeAttr(const CNodePtr &recv_node, const std::string &src_role) const {
  MS_EXCEPTION_IF_NULL(recv_node);

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
        inter_process_edges.push_back(src_role + std::to_string(i) + "->" + distributed::kEnvRoleOfPServer +
                                      std::to_string(rank_id_) + "_" + distributed::kEmbeddingCacheOps[k] + "_" +
                                      distributed::kParameterKey + std::to_string(param_key));
      }
    }
  }

  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRanks, MakeValue(src_ranks), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRoles, MakeValue(src_roles), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcNodeName, MakeValue(std::string(kEmbeddingLocalCacheNode)), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvDstNodeName, MakeValue(std::string(kEmbeddingRemoteCacheNode)), recv_node);

  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrIsMuxRpcKernel, MakeValue(true), recv_node);
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
  input_indices->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kInt32, std::make_shared<abstract::Shape>(kOneDimDynamicShape)));

  // 2. Create embedding lookup node.
  auto embedding_cache_lookup_node = CreateEmbeddingLookupKernel(graph, input_param, input_indices, node);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node);

  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), embedding_cache_lookup_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), embedding_cache_lookup_node);

  if (embedding_storage_manager.Exists(param_key)) {
    common::AnfAlgo::SetNodeAttr(kAttrEnableEmbeddingStorage, MakeValue(true), embedding_cache_lookup_node);
    common::AnfAlgo::SetNodeAttr(kAttrParameterKey, MakeValue(param_key), embedding_cache_lookup_node);
  }

  // 3. Create RpcSend node.
  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  send_inputs.push_back(embedding_cache_lookup_node);
  CNodePtr send_node = graph->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(send_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), send_node);
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
                                                                        const AnfNodePtr &node,
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
  input_indices->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kInt32, std::make_shared<abstract::Shape>(kOneDimDynamicShape)));

  ParameterPtr update_values = graph->add_parameter();
  MS_EXCEPTION_IF_NULL(update_values);
  auto emb_shape = common::AnfAlgo::GetOutputInferShape(param, 0);
  if (emb_shape.empty()) {
    MS_LOG(EXCEPTION) << "Embedding table shape is empty.";
  }
  ShapeVector update_values_shape = emb_shape;
  const int64_t dynamic_dim = -1;
  update_values_shape[0] = dynamic_dim;
  update_values->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kFloat32, std::make_shared<abstract::Shape>(update_values_shape)));

  // 2. Create embedding update node.
  auto embedding_cache_update_node = CreateEmbeddingUpdateKernel(graph, input_param, input_indices, update_values);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), embedding_cache_update_node);

  if (embedding_storage_manager.Exists(param_key)) {
    common::AnfAlgo::SetNodeAttr(kAttrEnableEmbeddingStorage, MakeValue(true), embedding_cache_update_node);
    common::AnfAlgo::SetNodeAttr(kAttrParameterKey, MakeValue(param_key), embedding_cache_update_node);
  }

  // 3. Create return node.
  CNodePtr return_node = CreateReturnNode(graph, embedding_cache_update_node);
  MS_EXCEPTION_IF_NULL(return_node);
  graph->set_return(return_node);

  MS_EXCEPTION_IF_NULL(root_graph_);
  auto manager = root_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(graph);
  return graph;
}

CNodePtr PsEmbeddingCacheInserter::CreateEmbeddingLookupKernel(const FuncGraphPtr &graph,
                                                               const ParameterPtr &input_param,
                                                               const ParameterPtr &input_indices,
                                                               const AnfNodePtr &origin_embedding_lookup_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_param);
  MS_EXCEPTION_IF_NULL(input_indices);
  MS_EXCEPTION_IF_NULL(origin_embedding_lookup_node);

  std::vector<AnfNodePtr> embedding_lookup_inputs;
  // Sparse format is true meaning embedding table implements in the form of hash, false means the form of tensor.
  if (!distributed::EmbeddingCacheTableManager::GetInstance().is_sparse_format()) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrOffset, dyn_cast<CNode>(origin_embedding_lookup_node))) {
      MS_LOG(EXCEPTION) << "Can not find offset attr of kernel: "
                        << origin_embedding_lookup_node->fullname_with_scope();
    }
    int64_t offset = common::AnfAlgo::GetNodeAttr<int64_t>(origin_embedding_lookup_node, kAttrOffset);
    ValueNodePtr offset_value_node = NewValueNode(offset);
    MS_EXCEPTION_IF_NULL(offset_value_node);

    PrimitivePtr embedding_lookup_primitive = std::make_shared<Primitive>(kEmbeddingLookupOpName);
    embedding_lookup_inputs = {NewValueNode(embedding_lookup_primitive), input_param, input_indices, offset_value_node};
  } else {
    PrimitivePtr embedding_lookup_primitive = std::make_shared<Primitive>(kMapTensorGetOpName);
    embedding_lookup_primitive->set_attr(kAttrInsertDefaultValue, MakeValue(false));
    embedding_lookup_inputs = {NewValueNode(embedding_lookup_primitive), input_param, input_indices};
  }

  return graph->NewCNode(embedding_lookup_inputs);
}

CNodePtr PsEmbeddingCacheInserter::CreateEmbeddingUpdateKernel(const FuncGraphPtr &graph,
                                                               const ParameterPtr &input_param,
                                                               const ParameterPtr &input_indices,
                                                               const ParameterPtr &update_values) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_param);
  MS_EXCEPTION_IF_NULL(input_indices);
  MS_EXCEPTION_IF_NULL(update_values);

  // Sparse format is true meaning embedding table implements in the form of hash, false means the form of tensor.
  bool is_sparse_format = distributed::EmbeddingCacheTableManager::GetInstance().is_sparse_format();
  PrimitivePtr embedding_update_primitive = is_sparse_format ? std::make_shared<Primitive>(kMapTensorPutOpName)
                                                             : std::make_shared<Primitive>(kScatterUpdateOpName);
  std::vector<AnfNodePtr> embedding_update_inputs{NewValueNode(embedding_update_primitive), input_param, input_indices,
                                                  update_values};
  return graph->NewCNode(embedding_update_inputs);
}

CNodePtr PsEmbeddingCacheInserter::CreateRecvNode() const {
  // 1. Create input parameter for RpcRecv node.
  // The indices input.
  MS_EXCEPTION_IF_NULL(root_graph_);
  ParameterPtr input_indices = root_graph_->add_parameter();
  MS_EXCEPTION_IF_NULL(input_indices);
  input_indices->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kInt32, std::make_shared<abstract::Shape>(kOneDimDynamicShape)));
  auto fake_input_indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, kOneDimShape);
  input_indices->set_default_param(fake_input_indices_tensor);

  // The update values input.
  ParameterPtr update_values = root_graph_->add_parameter();
  MS_EXCEPTION_IF_NULL(update_values);
  update_values->set_abstract(
    std::make_shared<abstract::AbstractTensor>(kFloat32, std::make_shared<abstract::Shape>(kTwoDimsDynamicShape)));
  auto fake_update_values_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, kTwoDimsShape);
  update_values->set_default_param(fake_update_values_tensor);

  // The service id input, used to choose service to execute.
  ParameterPtr service_id = root_graph_->add_parameter();
  MS_EXCEPTION_IF_NULL(service_id);
  service_id->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt32, kOneDimShape));
  auto fake_id_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, kOneDimDynamicShape);
  service_id->set_default_param(fake_id_tensor);

  // 2. Create a RpcRecv node.
  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  recv_inputs.push_back(input_indices);
  recv_inputs.push_back(update_values);
  recv_inputs.push_back(service_id);
  MS_EXCEPTION_IF_NULL(root_graph_);
  CNodePtr recv_node = root_graph_->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);

  SetRecvNodeAttr(recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), recv_node);

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
    AbstractBasePtrList lookup_partial_args_spec_list = {param->abstract(), recv_outputs[0]->abstract()};
    emb_lookup_partial_node->set_abstract(std::make_shared<abstract::PartialAbstractClosure>(
      emb_lookup_graph_value_abstract, lookup_partial_args_spec_list, emb_lookup_partial_node));

    make_tuple_inputs->push_back(emb_lookup_partial_node);

    // 2. Construct updating embedding service sub graph.
    auto update_emb_sub_graph = ConstructUpdateEmbeddingSubGraph(param, node, key);
    MS_EXCEPTION_IF_NULL(update_emb_sub_graph);
    auto update_emb_graph_value = NewValueNode(update_emb_sub_graph);
    MS_EXCEPTION_IF_NULL(update_emb_graph_value);
    auto update_emb_graph_value_abstract = std::make_shared<abstract::FuncGraphAbstractClosure>(
      update_emb_sub_graph, abstract::AnalysisContext::DummyContext());
    update_emb_graph_value->set_abstract(update_emb_graph_value_abstract);

    CNodePtr update_emb_partial_node = root_graph_->NewCNode(
      {NewValueNode(prim::kPrimPartial), update_emb_graph_value, param, recv_outputs[0], recv_outputs[1]});
    MS_EXCEPTION_IF_NULL(update_emb_partial_node);
    AbstractBasePtrList update_partial_args_spec_list = {param->abstract(), recv_outputs[0]->abstract(),
                                                         recv_outputs[1]->abstract()};
    update_emb_partial_node->set_abstract(std::make_shared<abstract::PartialAbstractClosure>(
      update_emb_graph_value_abstract, update_partial_args_spec_list, update_emb_partial_node));

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
  MS_EXCEPTION_IF_NULL(switch_layer_node);

  CNodePtr call_node = root_graph_->NewCNode({switch_layer_node});
  MS_EXCEPTION_IF_NULL(call_node);

  // 4. Replace origin output and useless nodes of origin function graph.
  AnfNodePtr old_output = root_graph_->output();
  AnfNodePtr new_output = CreateOutputNode(root_graph_, old_output);
  auto final_output_node = root_graph_->NewCNode({NewValueNode(prim::kPrimDepend), new_output, call_node});
  MS_EXCEPTION_IF_NULL(final_output_node);

  auto graph_manager = root_graph_->manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  return graph_manager->Replace(root_graph_->output(), final_output_node);
}

void PsEmbeddingCacheInserter::BuildEmbeddingStorages() {
  for (const auto &item : keys_to_params_) {
    int32_t key = item.first;
    ParameterPtr param = item.second;
    MS_EXCEPTION_IF_NULL(param);

    auto param_info = param->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    if (!param_info->use_persistent_storage()) {
      MS_LOG(INFO) << "No need to use embedding storage for this parameter(key): " << key;
      continue;
    }

    const std::vector<int64_t> &origin_shape = param_info->origin_shape();
    size_t origin_capacity = LongToSize(origin_shape.front());
    size_t origin_emb_dim = LongToSize(origin_shape.back());
    MS_LOG(INFO) << "Get a parameter for embedding storage: " << param->name() << ", origin emb_dim: " << origin_emb_dim
                 << ", origin capacity: " << origin_capacity;

    const std::vector<int64_t> &slice_shape = param_info->parameter_shape();
    if (slice_shape.size() != kEmbeddingTableDims) {
      MS_LOG(EXCEPTION)
        << "When build embedding storage, Embedding table should be 2 dims for embedding cache mode, but got: "
        << slice_shape.size() << " dims, param name: " << param->name() << ", param key: " << key;
    }
    size_t capacity = LongToSize(slice_shape.front());
    size_t emb_dim = LongToSize(slice_shape.back());

    auto shape = common::AnfAlgo::GetOutputInferShape(param, 0);
    auto iter = shapes_to_nodes_.find(shape);
    if (iter == shapes_to_nodes_.end()) {
      MS_LOG(EXCEPTION) << "Can not find cnode for parameter(key[" << key << "]) with shape: " << shape;
    }
    AnfNodePtr node = iter->second;
    const size_t param_index = 0;
    const size_t key_index = 1;

    TypeId key_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, key_index);
    TypeId param_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, param_index);
    // Create dense or sparse embedding storage and add into embedding storage manager.
    distributed::CreateEmbeddingStorage(std::make_pair(key_type, param_type), key, emb_dim, capacity);
    MS_LOG(INFO) << "Add a new embedding storage, key: " << key << ", emb_dim: " << emb_dim
                 << ", capacity: " << capacity << ", origin emb_dim:" << origin_emb_dim
                 << ", origin capacity: " << origin_capacity;
  }
}

bool PsEmbeddingCacheInserter::Run() {
  // Get EmbeddingLookup nodes which are executed on server from origin function graph.
  GetEmbeddingLookupNodes();

  // Get parameters enabled embedding cache of origin function graph.
  GetCacheEnableParameters();

  // Build embedding storages for parameters enabled embedding cache to read/write embedding table from/to persistent
  // storage.
  BuildEmbeddingStorages();

  // Construct the embedding cache graph of server.
  RETURN_IF_FALSE_WITH_LOG(ConstructEmbeddingCacheGraph(), "Construct embedding cache graph failed.");

  // Set attr(device target attr and graph split label) for all CNodes.
  SetAttrForAllNodes();

  MS_EXCEPTION_IF_NULL(root_graph_);
  // Need renormalize to infer shape and set abstract.
  root_graph_->set_flag(kFlagNeedRenormalize, true);
  return true;
}
}  // namespace parallel
}  // namespace mindspore
