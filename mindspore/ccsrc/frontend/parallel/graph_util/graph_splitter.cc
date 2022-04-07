/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/graph_splitter.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "include/common/utils/utils.h"
#include "base/core_ops.h"
#include "mindspore/core/utils/ms_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/draw.h"

namespace mindspore {
namespace parallel {
bool OperatorLabel::operator<(const OperatorLabel &label) const { return to_string() < label.to_string(); }

bool OperatorLabel::operator==(const OperatorLabel &label) const { return to_string() == label.to_string(); }

bool OperatorLabel::operator!=(const OperatorLabel &label) const { return !(*this == label); }

bool OperatorLabel::LooseEqual(const OperatorLabel &label) const {
  auto mode = distributed::DistExecutionMode::kPSMode;
  if (kLabelMatchingFuncMap.count(mode) == 0) {
    MS_LOG(ERROR) << "The mode " << mode << " is invalid.";
    return false;
  }
  return kLabelMatchingFuncMap.at(mode)(label, *this);
}

std::string OperatorLabel::to_string() const { return std::to_string(rank_id) + "_" + ms_role; }

ValueNodePtr CreateFakeValueNode(bool use_origin_node, const AnfNodePtr &origin_node) {
  tensor::TensorPtr fake_tensor = nullptr;
  if (use_origin_node) {
    MS_EXCEPTION_IF_NULL(origin_node);
    abstract::AbstractTensorPtr origin_abstract;
    if (origin_node->abstract()->isa<abstract::AbstractTuple>()) {
      // Defaultly, if the origin node's output is a tuple, get the abstract of the first element.
      auto get_one_tuple_element = origin_node->abstract()->cast<abstract::AbstractTuplePtr>()->elements()[0];
      origin_abstract = get_one_tuple_element->cast<abstract::AbstractTensorPtr>();
    } else {
      origin_abstract = origin_node->abstract()->cast<abstract::AbstractTensorPtr>();
    }
    MS_EXCEPTION_IF_NULL(origin_abstract);
    fake_tensor = std::make_shared<tensor::Tensor>(origin_abstract->element()->BuildType()->type_id(),
                                                   origin_abstract->shape()->shape());
    MS_EXCEPTION_IF_NULL(fake_tensor);
  } else {
    fake_tensor = std::make_shared<tensor::Tensor>(1.0);
    MS_EXCEPTION_IF_NULL(fake_tensor);
  }

  auto fake_value = NewValueNode(fake_tensor);
  MS_EXCEPTION_IF_NULL(fake_value);
  fake_value->set_abstract(fake_tensor->ToAbstract());
  return fake_value;
}

void SetSendNodeAttr(const AnfNodePtr &send_node, const InterProcessOpEdge &inter_process_edge) {
  const auto &send_src_node = inter_process_edge.src_node;
  const auto &send_dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(send_src_node);
  MS_EXCEPTION_IF_NULL(send_dst_node);
  MS_EXCEPTION_IF_NULL(send_node);

  std::string src_node_name = send_src_node->fullname_with_scope();
  std::string dst_node_name = send_dst_node->fullname_with_scope();

  // These attributes are the inter-process edge information.
  std::vector<uint32_t> dst_ranks = {inter_process_edge.dst_label.rank_id};
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRanks, MakeValue(dst_ranks), send_node);
  std::vector<std::string> dst_roles = {inter_process_edge.dst_label.ms_role};
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRoles, MakeValue(dst_roles), send_node);

  common::AnfAlgo::SetNodeAttr(kAttrSendSrcNodeName, MakeValue(src_node_name), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendDstNodeName, MakeValue(dst_node_name), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeName, MakeValue(inter_process_edge.to_string()), send_node);

  // Set send node to CPU for now.
  common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), send_node);
}

void SetRecvNodeAttr(const AnfNodePtr &recv_node, const InterProcessOpEdge &inter_process_edge) {
  const auto &recv_src_node = inter_process_edge.src_node;
  const auto &recv_dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(recv_src_node);
  MS_EXCEPTION_IF_NULL(recv_dst_node);
  MS_EXCEPTION_IF_NULL(recv_node);

  std::string src_node_name = recv_src_node->fullname_with_scope();
  std::string dst_node_name = recv_dst_node->fullname_with_scope();

  // These attributes are the inter-process edge information.
  std::vector<uint32_t> src_ranks = {inter_process_edge.src_label.rank_id};
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRanks, MakeValue(src_ranks), recv_node);
  std::vector<std::string> src_roles = {inter_process_edge.src_label.ms_role};
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRoles, MakeValue(src_roles), recv_node);

  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcNodeName, MakeValue(src_node_name), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvDstNodeName, MakeValue(dst_node_name), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeName, MakeValue(inter_process_edge.to_string()), recv_node);

  // Set recv node to CPU for now.
  common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), recv_node);
}

CNodePtr CreateSendNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge) {
  const auto &src_node = inter_process_edge.src_node;
  const auto &dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);

  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  ValueNodePtr mock_value = nullptr;
  if (IsPrimitiveCNode(src_node, prim::kPrimUpdateState)) {
    mock_value = CreateFakeValueNode(false);
    send_inputs.push_back(mock_value);
    send_inputs.push_back(src_node);
  } else {
    send_inputs.push_back(src_node);
    mock_value = CreateFakeValueNode(true, src_node);
  }
  CNodePtr send_node = func_graph->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(send_node);
  send_node->set_abstract(mock_value->abstract());

  SetSendNodeAttr(send_node, inter_process_edge);
  return send_node;
}

CNodePtr CreateRecvNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge) {
  const auto &src_node = inter_process_edge.src_node;
  const auto &dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);

  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  CNodePtr recv_node = nullptr;
  AbstractBasePtr recv_node_abs = nullptr;
  if (IsPrimitiveCNode(src_node, prim::kPrimUpdateState)) {
    ValuePtr monad_value = nullptr;
    if (HasAbstractUMonad(src_node)) {
      monad_value = kUMonad;
    } else if (HasAbstractIOMonad(src_node)) {
      monad_value = kIOMonad;
    } else {
      MS_LOG(EXCEPTION) << "The src_node is PrimUpdateState must have monad abstract.";
    }
    auto monad_input = NewValueNode(monad_value);
    MS_EXCEPTION_IF_NULL(monad_input);
    monad_input->set_abstract(monad_value->ToAbstract());
    recv_inputs.push_back(monad_input);
    recv_node_abs = src_node->abstract();
  } else {
    if (src_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrUpdateParameter, src_node->cast<CNodePtr>()) &&
        common::AnfAlgo::HasNodeAttr(kAttrParameterInputIndex, src_node->cast<CNodePtr>())) {
      int64_t parameter_index = common::AnfAlgo::GetNodeAttr<int64_t>(src_node, kAttrParameterInputIndex);
      auto kernel_with_index =
        common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(src_node->cast<CNodePtr>(), parameter_index), 0);
      auto param_node = kernel_with_index.first;
      recv_inputs.push_back(param_node);

      // To update the parameter on the device side in heterogeneous case, side-effect node should be added to recv's
      // input.
      ValuePtr monad_value = kUMonad;
      auto monad_input = NewValueNode(monad_value);
      MS_EXCEPTION_IF_NULL(monad_input);
      monad_input->set_abstract(monad_value->ToAbstract());
      recv_inputs.push_back(monad_input);

      recv_node_abs = param_node->abstract();
    } else {
      auto mock_value = CreateFakeValueNode(true, src_node);
      MS_EXCEPTION_IF_NULL(mock_value);
      recv_inputs.push_back(mock_value);
      recv_node_abs = src_node->abstract();
    }
  }
  recv_node = func_graph->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);
  recv_node->set_abstract(recv_node_abs);

  SetRecvNodeAttr(recv_node, inter_process_edge);
  return recv_node;
}

void ParameterServerMode::PreBuildDistributedGraph() {
  MS_LOG(INFO) << "Start pre-building distribtued graph in Parameter Server mode.";
  MS_EXCEPTION_IF_NULL(node_labels_);
  ProcessForSplitOptimizer();
  MS_LOG(INFO) << "End pre-building distribtued graph in Parameter Server mode.";
}

void ParameterServerMode::PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start post-building distribtued graph in Parameter Server mode.";
  MS_EXCEPTION_IF_NULL(node_labels_);
  // Judge the node role number validation.
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, worker number should be greater than 0.";
  }
  uint32_t server_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfServer);
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, server number should be greater than 0.";
  }
  // Only multiple worker scenario needs this optimizer.
  if (worker_num < kMinGradAccumWorkerNum) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> nodes = FuncGraph::TopoSort(return_node);
  std::vector<CNodePtr> ps_optimizer_node_list = FilterServerAwareOptimizerList(nodes);

  // Duplicate out degrees for ps optimizers because defaultly there's only one edge to the rank 0 worker.
  for (const auto &ps_optimizer : ps_optimizer_node_list) {
    for (const auto &edge_info : comm_edges) {
      if (edge_info.first.src_node == ps_optimizer) {
        // The optimizer's output should always connect to Send node which is the input of a MakeTuple node.
        // We need to replace the MakeTuple node with a new one.
        const auto &origin_send_node = std::get<0>(edge_info.second);
        std::vector<AnfNodePtr> new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), origin_send_node};
        AnfNodePtr dst_node = edge_info.first.dst_node;
        for (uint32_t i = 1; i < worker_num; i++) {
          OperatorLabel worker_label = {i, distributed::kEnvRoleOfWorker};
          InterProcessOpEdge edge = {ps_optimizer, node_labels_->at(ps_optimizer), dst_node, worker_label};
          auto duplicated_send_node = CreateSendNode(func_graph_, edge);
          node_labels_->insert(std::make_pair(duplicated_send_node, edge.src_label));
          new_make_tuple_inputs.emplace_back(duplicated_send_node);
        }
        auto new_make_tuple_node = func_graph_->NewCNode(new_make_tuple_inputs);
        new_make_tuple_node->set_abstract(new_make_tuple_inputs.back()->abstract());
        (void)func_graph_->manager()->Replace(origin_send_node, new_make_tuple_node);
      }
    }
  }
  MS_LOG(INFO) << "End post-building distribtued graph in Parameter Server mode.";
}

void ParameterServerMode::DoRpcNodeFusion() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  // Only the rpc nodes whose peer is the same process(with same OperatorLabel) can be fused.
  std::map<std::pair<OperatorLabel, std::string>, std::vector<CNodePtr>> rpc_nodes_list_need_to_be_fused;
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    std::string cnode_name = common::AnfAlgo::GetCNodeName(cnode);
    if (cnode_name != kRpcSendOpName && cnode_name != kRpcRecvOpName) {
      continue;
    }
    const auto &peer_ranks = (cnode_name == kRpcSendOpName)
                               ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrSendDstRanks)
                               : common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrRecvSrcRanks);
    const auto &peer_roles = (cnode_name == kRpcSendOpName)
                               ? common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(cnode, kAttrSendDstRoles)
                               : common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(cnode, kAttrRecvSrcRoles);
    OperatorLabel peer_label = {peer_ranks[0], peer_roles[0]};
    rpc_nodes_list_need_to_be_fused[std::make_pair(peer_label, cnode_name)].emplace_back(cnode);
  }

  for (auto &rpc_nodes_fuse_info : rpc_nodes_list_need_to_be_fused) {
    // Reorder the rpc nodes list according to the inter-process edge name so the inputs order of send/recv nodes can
    // correspond.
    std::sort(rpc_nodes_fuse_info.second.begin(), rpc_nodes_fuse_info.second.end(),
              [](const CNodePtr &a, const CNodePtr &b) {
                return common::AnfAlgo::GetNodeAttr<std::string>(a, kAttrInterProcessEdgeName) <
                       common::AnfAlgo::GetNodeAttr<std::string>(b, kAttrInterProcessEdgeName);
              });
    if (rpc_nodes_fuse_info.first.second == kRpcSendOpName) {
      FuseRpcSendNodes(rpc_nodes_fuse_info.second);
    } else {
      FuseRpcRecvNodes(rpc_nodes_fuse_info.second);
    }
  }
}

void ParameterServerMode::ProcessForSplitOptimizer() {
  // Judge the node role number validation.
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, worker number should be greater than 0.";
  }
  uint32_t server_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfServer);
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, server number should be greater than 0.";
  }
  // Only multiple worker scenario needs this optimizer.
  if (worker_num < kMinGradAccumWorkerNum) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> nodes = FuncGraph::TopoSort(return_node);
  std::vector<CNodePtr> ps_optimizer_node_list = FilterServerAwareOptimizerList(nodes);
  for (const auto &ps_optimizer : ps_optimizer_node_list) {
    MS_EXCEPTION_IF_NULL(ps_optimizer);

    // Load attributes for this optimizer.
    size_t gradient_index = common::AnfAlgo::HasNodeAttr(kAttrGradientInputIndex, ps_optimizer)
                              ? common::AnfAlgo::GetNodeAttr<int64_t>(ps_optimizer, kAttrGradientInputIndex)
                              : UINT64_MAX;
    size_t indices_index = common::AnfAlgo::HasNodeAttr(kAttrIndicesInputIndex, ps_optimizer)
                             ? common::AnfAlgo::GetNodeAttr<int64_t>(ps_optimizer, kAttrIndicesInputIndex)
                             : UINT64_MAX;
    std::string gradient_type = (common::AnfAlgo::HasNodeAttr(kAttrGradientType, ps_optimizer))
                                  ? common::AnfAlgo::GetNodeAttr<std::string>(ps_optimizer, kAttrGradientType)
                                  : kDenseGradient;
    if (kGradTypeToAccumOpName.count(gradient_type) == 0) {
      MS_LOG(EXCEPTION) << "The gradient type " << gradient_type << " is invalid.";
    }

    const std::string &opt_device_target = GetCNodeTarget(ps_optimizer);
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(ps_optimizer); i++) {
      auto input = common::AnfAlgo::GetInputNode(ps_optimizer, i);
      // If the input is not a cnode, no inter-process edge is added so no node with multiple inputs should be created.
      if (!input->isa<CNode>()) {
        continue;
      }

      if (i == gradient_index) {
        // Create the node to replace origin gradient which could be a RealDiv node.
        std::pair<CNodePtr, CNodePtr> grad_accum_nodes = CreateNodesForGradAccumulation(
          input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, gradient_type, worker_num);

        const auto &accum_node = grad_accum_nodes.first;
        const auto &real_div_node = grad_accum_nodes.second;
        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, real_div_node);
        node_labels_->insert(std::make_pair(accum_node, node_labels_->at(ps_optimizer)));
        node_labels_->insert(std::make_pair(real_div_node, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), accum_node);
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), real_div_node);
      } else if (i == indices_index) {
        // Create the node to replace origin indices.
        AnfNodePtr new_indices_input = CreateNodeWithInterProcessEdgeOnPServer(
          kConcatOpName, input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, worker_num);

        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, new_indices_input);
        node_labels_->insert(std::make_pair(new_indices_input, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), new_indices_input);
      } else {
        std::pair<CNodePtr, CNodePtr> make_tuple_get_item_nodes =
          CreateNodesForMakeTuple(input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, worker_num);

        auto &make_tuple_node = make_tuple_get_item_nodes.first;
        auto &tuple_get_item_node = make_tuple_get_item_nodes.second;
        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, tuple_get_item_node);
        node_labels_->insert(std::make_pair(make_tuple_node, node_labels_->at(ps_optimizer)));
        node_labels_->insert(std::make_pair(tuple_get_item_node, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), make_tuple_node);
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), tuple_get_item_node);
      }
    }
  }
}

std::vector<CNodePtr> ParameterServerMode::FilterServerAwareOptimizerList(const std::vector<AnfNodePtr> &nodes) {
  std::vector<CNodePtr> ps_optim_list = {};
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    if (common::AnfAlgo::HasNodeAttr(kAttrUpdateParameter, cnode)) {
      ps_optim_list.emplace_back(cnode);
    }
  }
  return ps_optim_list;
}

std::pair<CNodePtr, CNodePtr> ParameterServerMode::CreateNodesForGradAccumulation(const AnfNodePtr &gradient_input,
                                                                                  size_t gradient_input_index,
                                                                                  const std::string &gradient_type,
                                                                                  size_t total_gradient_number) {
  MS_EXCEPTION_IF_NULL(gradient_input);

  if (kGradTypeToAccumOpName.count(gradient_type) == 0) {
    MS_LOG(EXCEPTION) << "The gradient type " << gradient_type << " is invalid.";
  }
  const std::string &accum_node_name = kGradTypeToAccumOpName.at(gradient_type);
  CNodePtr grad_accum_node = CreateNodeWithInterProcessEdgeOnPServer(accum_node_name, gradient_input,
                                                                     gradient_input_index, total_gradient_number);
  MS_EXCEPTION_IF_NULL(grad_accum_node);

  CNodePtr grad_mean_node = CreateGradMeanNode(grad_accum_node, total_gradient_number);
  MS_EXCEPTION_IF_NULL(grad_mean_node);
  return std::make_pair(grad_accum_node, grad_mean_node);
}

CNodePtr ParameterServerMode::CreateGradMeanNode(const AnfNodePtr &gradient, size_t divisor) {
  MS_EXCEPTION_IF_NULL(gradient);

  // Step 1: Create the value node of divisor. The divisor's value is worker number.
  auto addn_abstract = gradient->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(addn_abstract);
  // Use reciprocal of the divisor so Mul node should be created.
  auto divisor_tensor =
    std::make_shared<tensor::Tensor>(1 / static_cast<double>(divisor), addn_abstract->element()->BuildType());
  MS_EXCEPTION_IF_NULL(divisor_tensor);
  auto divisor_value_node = NewValueNode(divisor_tensor);
  MS_EXCEPTION_IF_NULL(divisor_value_node);
  divisor_value_node->set_abstract(divisor_tensor->ToAbstract());

  // Step 2: Create Mul node.
  std::vector<AnfNodePtr> real_div_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), gradient,
                                             divisor_value_node};
  CNodePtr grad_mean_node = func_graph_->NewCNode(real_div_inputs);
  MS_EXCEPTION_IF_NULL(grad_mean_node);
  grad_mean_node->set_abstract(gradient->abstract());
  return grad_mean_node;
}

std::pair<CNodePtr, CNodePtr> ParameterServerMode::CreateNodesForMakeTuple(const AnfNodePtr &input, size_t input_index,
                                                                           size_t total_inputs_number) {
  MS_EXCEPTION_IF_NULL(input);
  CNodePtr make_tuple_node = CreateNodeWithInterProcessEdgeOnPServer(
    prim::kMakeTuple, input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, total_inputs_number);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  abstract::AbstractTuplePtr tuple_abstract = make_tuple_node->abstract()->cast<abstract::AbstractTuplePtr>();

  // For MakeTuple node on Parameter Server, we get the first input as its abstract because the other inputs are
  // supposed to be the same as the first one.
  size_t item_index = 0;
  auto item_index_value_node = NewValueNode(MakeValue(UlongToLong(item_index)));
  MS_EXCEPTION_IF_NULL(item_index_value_node);
  std::vector<AnfNodePtr> tuple_get_item_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                   make_tuple_node, item_index_value_node};
  CNodePtr tuple_get_item_node = func_graph_->NewCNode(tuple_get_item_inputs);
  MS_EXCEPTION_IF_NULL(tuple_get_item_node);
  tuple_get_item_node->set_abstract(tuple_abstract->elements()[0]);
  return std::make_pair(make_tuple_node, tuple_get_item_node);
}

CNodePtr ParameterServerMode::CreateNodeWithInterProcessEdgeOnPServer(const std::string &many_to_one_node_name,
                                                                      const AnfNodePtr &real_input,
                                                                      size_t index_of_real_input,
                                                                      uint32_t total_inputs_number) {
  if (index_of_real_input >= total_inputs_number) {
    MS_LOG(EXCEPTION) << "The index of real input for " << many_to_one_node_name << " " << index_of_real_input
                      << " is greater or equal to worker number " << total_inputs_number;
  }

  // Step 1: Create multiple inputs of new node including extra nodes.
  std::vector<AnfNodePtr> new_node_inputs;
  new_node_inputs.resize(total_inputs_number);
  std::vector<AnfNodePtr> mock_node_inputs = {NewValueNode(
    std::make_shared<Primitive>(IsPrimitiveCNode(real_input, prim::kPrimUpdateState) ? "UpdateState" : kVirtualNode))};
  for (size_t i = 0; i < new_node_inputs.size(); i++) {
    new_node_inputs[i] = func_graph_->NewCNode(mock_node_inputs);
    MS_EXCEPTION_IF_NULL(new_node_inputs[i]);
    new_node_inputs[i]->set_abstract(real_input->abstract());
    new_node_inputs[i]->cast<CNodePtr>()->set_fullname_with_scope(real_input->fullname_with_scope());

    // Set operator label for new node's inputs.
    OperatorLabel input_label = {SizeToUint(i), distributed::kEnvRoleOfWorker};
    node_labels_->insert(std::make_pair(new_node_inputs[i], input_label));
  }
  new_node_inputs[index_of_real_input] = real_input;

  // Step 2: Create the new node.
  auto new_node_prim = NewValueNode(std::make_shared<Primitive>(many_to_one_node_name));
  new_node_inputs.insert(new_node_inputs.begin(), new_node_prim);

  auto new_node = func_graph_->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);

  // Step 3: Set the new node's abstract and attrs.
  if (many_to_one_node_name == kAddNOpName) {
    common::AnfAlgo::SetNodeAttr("N", MakeValue(static_cast<int64_t>(total_inputs_number)), new_node);
    common::AnfAlgo::SetNodeAttr("n", MakeValue(static_cast<int64_t>(total_inputs_number)), new_node);
    new_node->set_abstract(real_input->abstract());
  } else if (many_to_one_node_name == kConcatOpName) {
    auto origin_abs = real_input->abstract()->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(origin_abs);

    auto new_abs = origin_abs->Clone()->cast<abstract::AbstractTensorPtr>();
    ShapeVector new_shape = new_abs->shape()->shape();
    new_shape[0] = new_shape[0] * total_inputs_number;
    new_abs->shape()->set_shape(new_shape);
    new_node->set_abstract(new_abs);

    // Concat node must have attribute "axis" or kernel building will fail.
    size_t axis_index = 0;
    common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(UlongToLong(axis_index)), new_node);
  } else if (many_to_one_node_name == prim::kMakeTuple) {
    AbstractBasePtrList abstract_list = {};
    auto first_input = new_node_inputs.begin();
    std::advance(first_input, 1);
    (void)std::for_each(first_input, new_node_inputs.end(),
                        [&](const auto &input) { abstract_list.emplace_back(input->abstract()); });
    new_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  } else {
    new_node->set_abstract(real_input->abstract());
  }
  return new_node;
}

bool ParameterServerMode::FuseRpcSendNodes(const std::vector<CNodePtr> &rpc_send_nodes) {
  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  AbstractBasePtrList abstract_list = {};
  std::string fused_inter_process_edge_name = "";
  for (const auto &send_node : rpc_send_nodes) {
    MS_EXCEPTION_IF_NULL(send_node);
    for (size_t i = 1; i < send_node->inputs().size(); i++) {
      auto input_i = send_node->inputs()[i];
      MS_EXCEPTION_IF_NULL(input_i);
      // If the input of send is monad, do not pass it to fused send node.
      if (HasAbstractMonad(input_i)) {
        continue;
      }
      send_inputs.emplace_back(input_i);
    }
    abstract_list.emplace_back(send_node->abstract());
    fused_inter_process_edge_name.append(
      common::AnfAlgo::GetNodeAttr<std::string>(send_node, kAttrInterProcessEdgeName));
  }

  CNodePtr fused_send_node = func_graph_->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(fused_send_node);
  fused_send_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeName, MakeValue(fused_inter_process_edge_name), fused_send_node);

  for (size_t j = 0; j < rpc_send_nodes.size(); j++) {
    auto index_node = NewValueNode(MakeValue(SizeToLong(j)));
    MS_EXCEPTION_IF_NULL(index_node);
    std::vector<AnfNodePtr> tuple_get_item_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                     fused_send_node, index_node};
    CNodePtr tuple_get_item_node = func_graph_->NewCNode(tuple_get_item_inputs);
    MS_EXCEPTION_IF_NULL(tuple_get_item_node);
    tuple_get_item_node->set_abstract(abstract_list[j]);
    func_graph_->manager()->Replace(rpc_send_nodes[j], tuple_get_item_node);
  }
  return true;
}

bool ParameterServerMode::FuseRpcRecvNodes(const std::vector<CNodePtr> &rpc_recv_nodes) {
  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  AbstractBasePtrList abstract_list = {};
  std::string fused_inter_process_edge_name = "";
  for (const auto &recv_node : rpc_recv_nodes) {
    MS_EXCEPTION_IF_NULL(recv_node);
    for (size_t i = 1; i < recv_node->inputs().size(); i++) {
      auto input_i = recv_node->inputs()[i];
      MS_EXCEPTION_IF_NULL(input_i);
      recv_inputs.emplace_back(input_i);
    }
    abstract_list.emplace_back(recv_node->abstract());
    fused_inter_process_edge_name.append(
      common::AnfAlgo::GetNodeAttr<std::string>(recv_node, kAttrInterProcessEdgeName));
  }

  CNodePtr fused_recv_node = func_graph_->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(fused_recv_node);
  fused_recv_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeName, MakeValue(fused_inter_process_edge_name), fused_recv_node);

  for (size_t j = 0; j < rpc_recv_nodes.size(); j++) {
    auto index_node = NewValueNode(MakeValue(SizeToLong(j)));
    MS_EXCEPTION_IF_NULL(index_node);
    std::vector<AnfNodePtr> tuple_get_item_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                     fused_recv_node, index_node};
    CNodePtr tuple_get_item_node = func_graph_->NewCNode(tuple_get_item_inputs);
    MS_EXCEPTION_IF_NULL(tuple_get_item_node);
    tuple_get_item_node->set_abstract(abstract_list[j]);
    func_graph_->manager()->Replace(rpc_recv_nodes[j], tuple_get_item_node);
  }
  return true;
}

GraphSplitter::GraphSplitter(const FuncGraphPtr &func_graph, uint32_t rank_id, const std::string &role)
    : func_graph_(func_graph),
      rank_id_(rank_id),
      role_(role),
      mode_(distributed::DistExecutionMode::kPSMode),
      exec_mode_(nullptr),
      this_process_label_({rank_id, role}) {
  default_label_ = {0, distributed::kEnvRoleOfWorker};
}

GraphSplitter::~GraphSplitter() { node_labels_.clear(); }

void GraphSplitter::Run() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_->manager());

  // Step 1: Dye all the nodes of the whole func_graph_.
  DyeGraph();
  // If all nodes are all on this process, no need to split the graph. So return.
  if (std::find_if(node_labels_.begin(), node_labels_.end(), [&](const auto &node_to_label) {
        return node_to_label.second != this_process_label_;
      }) == node_labels_.end()) {
    MS_LOG(INFO) << "No need to build and split distributed graph.";
    return;
  }

  // Step 2: Create exec_mode_ according to the current execution mode.
  CreateExecutionMode();

  // Step 3: Prebuild the distributed graph before it gets split.
  exec_mode_->PreBuildDistributedGraph();

  // Step 4: Generate the node segments with different labels.
  std::vector<SplitGraphSegment> segments = GenerateSplitSegments();
  // If the segment number is 0, there will be no distributed execution.
  if (segments.empty()) {
    return;
  }

  // Step 5: Create inter-process operators for segments with different labels.
  InterProcessOpEdgesInfo comm_edges = GenerateInterProcessOperators();

  // Step 6: Split the graph and eliminate extra nodes.
  SplitGraph(segments, comm_edges);

  // Step 7: Postbuild the graph after splitting.
  exec_mode_->PostBuildDistributedGraph(comm_edges);

  // Step 8: Fuse the rpc nodes to improve performance.
  exec_mode_->DoRpcNodeFusion();
}

void GraphSplitter::DyeGraph() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Mark all nodes with original label at the beginning. This means the node is supposed to be on the process with
    // default_label_.
    node_labels_[node] = default_label_;
    if (node->isa<CNode>()) {
      // For CNodes, mark them with the label passed by frontend if has one.
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      OperatorLabel label = GetSplitLabel(cnode);
      node_labels_[node] = label;
    }

    // If the node's label is the same as this process's, set its label to this_process_label_.
    if (this_process_label_.LooseEqual(node_labels_[node])) {
      node_labels_[node] = this_process_label_;
    }
  });
}

void GraphSplitter::CreateExecutionMode() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (node_labels_.empty()) {
    MS_LOG(EXCEPTION) << "Must dye the original graph before creating execution mode.";
  }
  if (mode_ == distributed::DistExecutionMode::kPSMode) {
    exec_mode_ = std::make_unique<ParameterServerMode>(func_graph_, &node_labels_, rank_id_, role_);
  }
  MS_EXCEPTION_IF_NULL(exec_mode_);
}

std::vector<SplitGraphSegment> GraphSplitter::GenerateSplitSegments() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> nodes = FuncGraph::TopoSort(return_node);

  std::vector<SplitGraphSegment> results = {};
  SplitGraphSegment segment;
  OperatorLabel last_label = this_process_label_;
  segment.label = last_label;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto cnode_split_label = node_labels_[n];
    // If this node's label is not the same as last node's, create a segment from 'segment_nodes'.
    if (cnode_split_label != last_label && !segment.nodes.empty()) {
      (void)results.emplace_back(segment);
      segment.nodes.clear();
    }
    // Mark the last label.
    last_label = cnode_split_label;
    segment.label = cnode_split_label;
    (void)segment.nodes.emplace_back(n);
  }

  // Add the last segment.
  (void)results.emplace_back(segment);
  MS_LOG(INFO) << "Segments number with different distributed split labels is " << results.size();
  return results;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOperators() {
  InterProcessOpEdgesInfo comm_edges = {};
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    // Only support to split CNode to other process.
    if (!node->isa<CNode>()) {
      continue;
    }

    // Generating send/recv nodes for each nodes' inputs will be enough.
    auto node_inputs_comm_edges = GenerateInterProcessOpsForNodeInputs(node);
    comm_edges.insert(node_inputs_comm_edges.begin(), node_inputs_comm_edges.end());
  }
  MS_LOG(INFO) << "The communication edge number is " << comm_edges.size();
  return comm_edges;
}

void GraphSplitter::SplitGraph(const std::vector<SplitGraphSegment> &segments,
                               const InterProcessOpEdgesInfo &comm_edges) {
  // Step 1: Traverse all the segments to add Depend for this process's graph.
  // The list of corresponding in and out degrees. In another word, the map between one segments' input send nodes. and
  // output recv nodes.
  InOutDegreeList in_out_degree_list = GenerateInOutDegreeList(segments, comm_edges);
  if (in_out_degree_list.empty()) {
    MS_LOG(WARNING) << "After splitting, this process has no graph on it. So optimize out the whole graph.";
    auto return_value_node = CreateFakeValueNode(false);
    (void)func_graph_->manager()->Replace(func_graph_->output(), return_value_node);
    return;
  }

  // Step 2: Add dependency between segments on this process.
  AddDependencyBetweenSegments(in_out_degree_list);

  // Step 3: Eliminate nodes not on this process.
  EliminateExtraNodes(comm_edges);
}

void GraphSplitter::DumpDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &edge : comm_edges) {
    auto send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    auto user_node_index = std::get<3>(send_recv_pair);
    func_graph_->manager()->SetEdge(recv_node, 1, send_node);
    func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
  }
  MS_LOG(INFO) << "Cut graph without eliminating nodes.";
  draw::Draw("single_node_graph.dot", func_graph_);
}

OperatorLabel GraphSplitter::GetSplitLabel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only CNode has distributed split label.";
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  auto prim_node = cnode->input(0);
  if (IsValueNode<Primitive>(prim_node)) {
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole)) {
      MS_LOG(INFO) << "CNode which has distributed split label: " << cnode->fullname_with_scope();
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(prim->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(prim->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  }
  return default_label_;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOpsForNodeInputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  InterProcessOpEdgesInfo comm_edges = {};
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_i = cnode->inputs()[i];
    MS_EXCEPTION_IF_NULL(input_i);

    // If the input's not a cnode, or its label is the same as this node's, there's no need to add communication nodes.
    if (!input_i->isa<CNode>() || IsNodesWithSameLabel(input_i, cnode)) {
      continue;
    }

    InterProcessOpEdge edge = {input_i, node_labels_[input_i], cnode, node_labels_[cnode]};

    auto send_node = CreateSendNode(func_graph_, edge);
    MS_EXCEPTION_IF_NULL(send_node);
    // The label should be the same as the node which will 'launch' Send node.
    node_labels_[send_node] = edge.src_label;

    auto recv_node = CreateRecvNode(func_graph_, edge);
    MS_EXCEPTION_IF_NULL(recv_node);
    // The label should be the same as the node which Receives the 'input'.
    node_labels_[recv_node] = edge.dst_label;

    auto comm_node_pair = std::make_tuple(send_node, recv_node, cnode, SizeToInt(i));
    (void)comm_edges.insert(std::make_pair(edge, comm_node_pair));
  }
  return comm_edges;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                                const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results = {};
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto input_i = cnode->inputs()[i];
      InterProcessOpEdge edge = {input_i, node_labels_[input_i], cnode, node_labels_[cnode]};
      if (comm_edges.count(edge) != 0 && edge.src_label == this_process_label_) {
        MS_LOG(INFO) << edge.to_string() << " is a communication edge.";
        auto comm_node_pair = comm_edges.at(edge);
        (void)results.emplace_back(std::get<0>(comm_node_pair));
      }
    }
  }
  return results;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                                 const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results = {};
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    auto users = func_graph_->manager()->node_users()[cnode];
    for (auto &u : users) {
      auto user_node = u.first->cast<CNodePtr>();
      InterProcessOpEdge edge = {cnode, node_labels_[cnode], user_node, node_labels_[user_node]};
      if (comm_edges.count(edge) != 0 && edge.dst_label == this_process_label_) {
        MS_LOG(INFO) << edge.to_string() << " is a communication edge.";
        auto comm_node_pair = comm_edges.at(edge);
        (void)results.emplace_back(std::get<1>(comm_node_pair));
      }
    }
  }
  return results;
}

InOutDegreeList GraphSplitter::GenerateInOutDegreeList(const std::vector<SplitGraphSegment> &segments,
                                                       const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start finding inter-process in-degrees.";

  InOutDegreeList in_out_degree_list;
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &segment : segments) {
    // If this segment should be on current process, continue.
    if (segment.label == this_process_label_) {
      continue;
    }
    std::vector<AnfNodePtr> nodes = segment.nodes;
    if (nodes.empty()) {
      MS_LOG(EXCEPTION) << "This segment is empty.";
      return in_out_degree_list;
    }

    auto segment_first_node = nodes[0];
    if (node_labels_[segment_first_node] != segment.label) {
      MS_LOG(EXCEPTION) << "Node label " << node_labels_[segment_first_node].to_string()
                        << " is not the same as segment label " << segment.label.to_string();
    }

    // Prepare for adding Depend between in-degree and out-degree of this segment because the execution order should be
    // kept consistent.
    std::vector<AnfNodePtr> concerned_in_degree_nodes = FindInterProcessInDegree(nodes, comm_edges);
    std::vector<AnfNodePtr> concerned_out_degree_nodes = FindInterProcessOutDegree(nodes, comm_edges);
    if (concerned_in_degree_nodes.empty()) {
      continue;
    }
    in_out_degree_list.emplace_back(std::make_pair(concerned_in_degree_nodes, concerned_out_degree_nodes));
  }
  MS_LOG(INFO) << "End finding inter-process in-degrees.";
  return in_out_degree_list;
}

void GraphSplitter::AddDependencyBetweenSegments(const InOutDegreeList &in_out_degree_list) {
  MS_LOG(INFO) << "Start adding dependency between segments.";
  // This tuple is key to the dependency of send nodes so that they will not be optimized out in some cases.
  std::vector<AnfNodePtr> send_node_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < in_out_degree_list.size(); i++) {
    auto &concerned_in_degree_nodes = in_out_degree_list[i].first;
    auto &concerned_out_degree_nodes = in_out_degree_list[i].second;
    send_node_tuple_inputs.insert(send_node_tuple_inputs.end(), concerned_in_degree_nodes.begin(),
                                  concerned_in_degree_nodes.end());
    if (concerned_out_degree_nodes.empty()) {
      // If this is the last segment's in and out degrees and has no out degrees, connect the send nodes to graph's
      // output.
      if (i == in_out_degree_list.size() - 1) {
        auto make_tuple_node = func_graph_->NewCNode(send_node_tuple_inputs);
        std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend)};
        out.push_back(send_node_tuple_inputs.back());
        out.push_back(make_tuple_node);
        auto out_node = func_graph_->NewCNode(out);
        MS_EXCEPTION_IF_NULL(out_node);
        out_node->set_abstract(send_node_tuple_inputs.back()->abstract());
        (void)func_graph_->manager()->Replace(func_graph_->output(), out_node);
      }
    } else {
      auto make_tuple_node = func_graph_->NewCNode(send_node_tuple_inputs);
      for (auto &recv : concerned_out_degree_nodes) {
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), recv->cast<CNodePtr>()->inputs()[1],
                                                make_tuple_node};
        auto depend = func_graph_->NewCNode(depend_input);
        depend->set_abstract(recv->cast<CNodePtr>()->inputs()[1]->abstract());
        func_graph_->manager()->SetEdge(recv, 1, depend);
      }
      // Reset the make tuple node inputs for next segments in degrees.
      send_node_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    }
  }
  MS_LOG(INFO) << "End adding dependency between segments.";
}

void GraphSplitter::EliminateExtraNodes(const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start eliminating nodes not on this process.";
  // Eliminate nodes which should be launched by other processes by set output edge.
  for (auto &edge : comm_edges) {
    InterProcessOpPair send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    int user_node_index = std::get<3>(send_recv_pair);

    OperatorLabel send_label = node_labels_[send_node];
    OperatorLabel recv_label = node_labels_[recv_node];
    if (send_label == recv_label) {
      MS_LOG(EXCEPTION) << "The Send and Recv must have different label. But got Send: " << send_label.to_string()
                        << ", Recv: " << recv_label.to_string();
    }

    if (recv_label == this_process_label_) {
      func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
    }
  }
  MS_LOG(INFO) << "End eliminating nodes not on this process.";
}

bool GraphSplitter::IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  if (node_labels_.count(node1) == 0 || node_labels_.count(node2) == 0) {
    MS_LOG(EXCEPTION) << "Either 'node1': " << node1->fullname_with_scope()
                      << " or 'node2': " << node2->fullname_with_scope() << " is not marked with split label.";
  }
  return node_labels_[node1] == node_labels_[node2];
}
}  // namespace parallel
}  // namespace mindspore
