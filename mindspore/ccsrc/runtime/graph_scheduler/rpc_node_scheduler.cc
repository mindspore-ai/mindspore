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

#include "runtime/graph_scheduler/rpc_node_scheduler.h"
#include <vector>
#include <string>
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/graph_scheduler/actor/rpc/mux_send_actor.h"
#include "runtime/graph_scheduler/actor/rpc/mux_recv_actor.h"

namespace mindspore {
namespace runtime {
namespace {
// MuxSendActor and MuxRecvActor of the server are used in pairs, and the MuxSendActor
// needs to obtain the information(ip and port) of peer that initiates this service from the corresponding
// MuxRecvActor to response request, so need to set the paired MuxRecvActor for MuxSendActor.
void SetMuxRecvActorForMuxSendActor(const RpcActorSetPtr &rpc_actor_set) {
  MS_EXCEPTION_IF_NULL(rpc_actor_set);

  // 1. Check whether exist mux recv actor.
  bool exist_mux_recv_actor = false;
  std::vector<RecvActorPtr> recv_actors;
  for (const auto &recv_actor : rpc_actor_set->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    CNodePtr rpc_recv_kernel = recv_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_recv_kernel);
    if (common::AnfAlgo::HasNodeAttr(kAttrIsMuxRpcKernel, rpc_recv_kernel) &&
        (common::AnfAlgo::GetNodeAttr<bool>(rpc_recv_kernel, kAttrIsMuxRpcKernel) == true)) {
      exist_mux_recv_actor = true;
      (void)recv_actors.emplace_back(recv_actor);
    }
  }

  if (!exist_mux_recv_actor) {
    return;
  }
  if (recv_actors.size() != 1) {
    MS_LOG(EXCEPTION) << "Currently the actor set is only allowed to contain one MuxRecvActor, but got: "
                      << recv_actors.size();
  }

  // 2. Set mux recv actor for mux send actor.
  MuxRecvActorPtr mux_recv_actor = std::dynamic_pointer_cast<MuxRecvActor>(recv_actors.front());
  MS_EXCEPTION_IF_NULL(mux_recv_actor);

  for (const auto &send_actor : rpc_actor_set->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    MuxSendActorPtr mux_send_actor = std::dynamic_pointer_cast<MuxSendActor>(send_actor);
    MS_EXCEPTION_IF_NULL(mux_send_actor);
    mux_send_actor->set_mux_recv_actor(mux_recv_actor);
  }
}
}  // namespace

RpcActorSetPtr RpcNodeScheduler::Build(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);

  // RpcActor inherits from KernelActor, so we need to filter out the rpc actors from kernel actors list.
  std::vector<KernelActorPtr> kernel_actors = actor_set->kernel_actors_;
  RpcActorSetPtr rpc_actor_set = std::make_shared<RpcActorSet>();
  MS_EXCEPTION_IF_NULL(rpc_actor_set);

  std::vector<RpcActorPtr> rpc_actors;
  for (const auto &kernel_actor : kernel_actors) {
    auto rpc_actor = std::dynamic_pointer_cast<RpcActor>(kernel_actor);
    if (std::dynamic_pointer_cast<RpcActor>(kernel_actor) == nullptr) {
      continue;
    } else {
      (void)rpc_actors.emplace_back(rpc_actor);
      if (std::dynamic_pointer_cast<SendActor>(rpc_actor) != nullptr) {
        (void)rpc_actor_set->send_actors_.emplace_back(std::dynamic_pointer_cast<SendActor>(rpc_actor));
      } else if (std::dynamic_pointer_cast<RecvActor>(rpc_actor) != nullptr) {
        (void)rpc_actor_set->recv_actors_.emplace_back(std::dynamic_pointer_cast<RecvActor>(rpc_actor));
      } else {
        MS_LOG(EXCEPTION) << "Rpc actor should be either SendActor or RecvActor.";
      }
    }
  }

  // Set the paired MuxRecvActor for MuxSendActor, used in embedding cache case.
  SetMuxRecvActorForMuxSendActor(rpc_actor_set);

  // Create route table proxy for each rpc actor and set.
  for (auto &rpc_actor : rpc_actors) {
    auto proxy = CreateRouteTableProxy();
    MS_EXCEPTION_IF_NULL(rpc_actor);
    MS_EXCEPTION_IF_NULL(proxy);
    rpc_actor->set_actor_route_table_proxy(proxy);
  }

  // Update the reference counts of rpc kernel inputs and workspaces.
  UpdateRpcActorRefCounts(rpc_actor_set);

  return rpc_actor_set;
}

void RpcNodeScheduler::Link(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  RpcActorSetPtr rpc_actor_set = actor_set->rpc_actors_;
  MS_EXCEPTION_IF_NULL(rpc_actor_set);
  std::vector<SendActorPtr> send_actors = rpc_actor_set->send_actors_;
  std::vector<RecvActorPtr> recv_actors = rpc_actor_set->recv_actors_;

  // The inter-process edge is connected to a remote peer. So the peer info attributes in the kernel should be
  // sufficient for route table.
  for (auto &send_actor : send_actors) {
    MS_EXCEPTION_IF_NULL(send_actor);
    CNodePtr rpc_send_kernel = send_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_send_kernel);

    auto send_dst_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_send_kernel, kAttrSendDstRanks);
    auto send_dst_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_send_kernel, kAttrSendDstRoles);
    std::string send_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendSrcNodeName);
    std::string send_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendDstNodeName);
    std::vector<std::string> edge_names =
      common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_send_kernel, kAttrInterProcessEdgeNames);

    if (send_dst_ranks.empty() || send_dst_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of send node " << rpc_send_kernel->fullname_with_scope()
                        << " is invalid. send_dst_ranks: " << send_dst_ranks << ", send_dst_roles: " << send_dst_roles
                        << ", send_src_node_name: " << send_src_node_name
                        << ", send_dst_node_name: " << send_dst_node_name;
    }
    send_actor->set_inter_process_edge_names(edge_names);
    send_actor->SetRouteInfo(send_dst_ranks[0], send_dst_roles[0], send_src_node_name, send_dst_node_name);
  }
  for (auto &recv_actor : recv_actors) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    CNodePtr rpc_recv_kernel = recv_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_recv_kernel);

    auto recv_src_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_recv_kernel, kAttrRecvSrcRanks);
    auto recv_src_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_recv_kernel, kAttrRecvSrcRoles);
    std::string recv_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvSrcNodeName);
    std::string recv_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvDstNodeName);
    std::vector<std::string> edge_names =
      common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_recv_kernel, kAttrInterProcessEdgeNames);

    if (recv_src_ranks.empty() || recv_src_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of recv node " << rpc_recv_kernel->fullname_with_scope()
                        << " is invalid. recv_src_ranks: " << recv_src_ranks << ", recv_src_roles: " << recv_src_roles
                        << ", recv_src_node_name: " << recv_src_node_name
                        << ", recv_dst_node_name: " << recv_dst_node_name;
    }
    recv_actor->set_inter_process_edge_names(edge_names);
    recv_actor->SetRouteInfo(recv_src_ranks[0], recv_src_roles[0], recv_src_node_name, recv_dst_node_name);
  }
}

void RpcNodeScheduler::Schedule(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  RpcActorSetPtr rpc_actor_set = actor_set->rpc_actors_;
  MS_EXCEPTION_IF_NULL(rpc_actor_set);
  // Must start server and register route table before looking up route and connecting.

  // Start servers of recv actors and register route table.
  for (auto &recv_actor : rpc_actor_set->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    if (!recv_actor->StartServer()) {
      MS_LOG(EXCEPTION) << "Failed to start server for the recv actor.";
    }
  }
  // Lookup route and connect to servers for send actors.
  for (auto &send_actor : rpc_actor_set->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    if (!send_actor->ConnectServer()) {
      MS_LOG(EXCEPTION) << "Failed to connect servers for the send actor.";
    }
  }
}

void RpcNodeScheduler::SetOpcontext(const RpcActorSetPtr &rpc_actors, OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(rpc_actors);

  for (auto &recv_actor : rpc_actors->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    recv_actor->SetOpcontext(op_context);
  }
  for (auto &send_actor : rpc_actors->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    send_actor->SetOpcontext(op_context);
  }

  // Set op_context and rpc actor set for later usage.
  op_context_ = op_context;
  rpc_actors_ = rpc_actors;
}

void RpcNodeScheduler::ResetOpcontext(const RpcActorSetPtr &rpc_actors) {
  MS_EXCEPTION_IF_NULL(rpc_actors);

  for (auto &recv_actor : rpc_actors->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    recv_actor->ResetOpcontext();
  }
  for (auto &send_actor : rpc_actors->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    send_actor->ResetOpcontext();
  }
  op_context_ = nullptr;
}

void RpcNodeScheduler::Clear() {
  if (rpc_actors_ != nullptr) {
    MS_LOG(INFO) << "Start finalizing tcp server and client for rpc actors.";
    for (auto &recv_actor : rpc_actors_->recv_actors_) {
      MS_EXCEPTION_IF_NULL(recv_actor);
      recv_actor->Clear();
    }
    for (auto &send_actor : rpc_actors_->send_actors_) {
      MS_EXCEPTION_IF_NULL(send_actor);
      send_actor->Clear();
    }
    MS_LOG(INFO) << "End finalizing tcp server and client for rpc actors.";
  }
}

void RpcNodeScheduler::Abort() {
  MS_LOG(INFO) << "Start aborting rpc actors.";
  MS_ERROR_IF_NULL_WO_RET_VAL(rpc_actors_);
  for (const auto &recv_actor : rpc_actors_->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    ActorDispatcher::Send(recv_actor->GetAID(), &RecvActor::StopRpcAtException);
  }
  MS_LOG(INFO) << "End aborting rpc actors.";

  if (op_context_ != nullptr) {
    // Set op_context success to exit output actor.
    SET_OPCONTEXT_SUCCESS_RET(*op_context_);
  }
}

void RpcNodeScheduler::UpdateRpcActorRefCounts(RpcActorSetPtr rpc_actor_set) const {
  MS_EXCEPTION_IF_NULL(rpc_actor_set);
  for (const auto &send_actor : rpc_actor_set->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    auto kernel_mod = AnfAlgo::GetKernelMod(send_actor->kernel_);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    size_t workspace_num = kernel_mod->GetWorkspaceSizeList().size();
    if (workspace_num == 0) {
      MS_LOG(EXCEPTION) << "Rpc send kernel must have workspace assigned.";
    }
    for (size_t i = 0; i < workspace_num; ++i) {
      auto device_tensor = AnfAlgo::GetMutableWorkspaceAddr(send_actor->kernel_, i);
      MS_EXCEPTION_IF_NULL(device_tensor);
      UpdateRefCount(device_tensor.get());
    }
  }
}

ActorRouteTableProxyPtr RpcNodeScheduler::CreateRouteTableProxy() const {
  ActorRouteTableProxyPtr actor_route_table_proxy;
  if (!ClusterContext::instance()->IsScheduler()) {
    auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
      ClusterContext::instance()->node_base());
    actor_route_table_proxy = std::make_shared<ActorRouteTableProxy>(cgn);
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy);
  }
  return actor_route_table_proxy;
}

RpcActorStatusUpdater &RpcActorStatusUpdater::GetInstance() {
  static RpcActorStatusUpdater instance;
  return instance;
}

void RpcActorStatusUpdater::set_rpc_actors(const std::string &graph_name, const RpcActorSetPtr &rpc_actors) {
  if (rpc_actors != nullptr) {
    graph_to_rpc_actors_[graph_name] = rpc_actors;
  }
}

void RpcActorStatusUpdater::UpdateRpcActorStatus(const std::string &graph_name) {
  // Update status for recv actors to control their execution orders.
  if (graph_to_rpc_actors_.count(graph_name) != 0) {
    auto rpc_actors = graph_to_rpc_actors_[graph_name];
    if (rpc_actors.lock() != nullptr) {
      for (auto &recv_actor : rpc_actors.lock()->recv_actors_) {
        MS_EXCEPTION_IF_NULL(recv_actor);
        recv_actor->UpdateStatus();
      }
    }
  }
}

void RpcActorStatusUpdater::FlushRpcData(const std::string &graph_name) {
  // Flush data for send actors.
  if (graph_to_rpc_actors_.count(graph_name) != 0) {
    auto rpc_actors = graph_to_rpc_actors_[graph_name];
    if (rpc_actors.lock() != nullptr) {
      for (auto &send_actor : rpc_actors.lock()->send_actors_) {
        MS_EXCEPTION_IF_NULL(send_actor);
        send_actor->FlushData();
      }
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
