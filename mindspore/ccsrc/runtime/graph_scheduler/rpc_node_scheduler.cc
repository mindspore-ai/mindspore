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
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace runtime {
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
      rpc_actors.emplace_back(rpc_actor);
      if (std::dynamic_pointer_cast<SendActor>(rpc_actor) != nullptr) {
        rpc_actor_set->send_actors_.emplace_back(std::dynamic_pointer_cast<SendActor>(rpc_actor));
      } else if (std::dynamic_pointer_cast<RecvActor>(rpc_actor) != nullptr) {
        rpc_actor_set->recv_actors_.emplace_back(std::dynamic_pointer_cast<RecvActor>(rpc_actor));
      } else {
        MS_LOG(EXCEPTION) << "Rpc actor should be either SendActor or RecvActor.";
      }
    }
  }

  // Create route table proxy for each rpc actor and set.
  for (auto &rpc_actor : rpc_actors) {
    auto proxy = CreateRouteTableProxy();
    MS_EXCEPTION_IF_NULL(proxy);
    rpc_actor->set_actor_route_table_proxy(proxy);
  }

  return rpc_actor_set;
}

void RpcNodeScheduler::Link(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  RpcActorSetPtr rpc_actor_set = actor_set->rpc_actors_;
  MS_EXCEPTION_IF_NULL(rpc_actor_set);
  std::vector<SendActorPtr> send_actors = rpc_actor_set->send_actors_;
  std::vector<RecvActorPtr> recv_actors = rpc_actor_set->recv_actors_;

  // The inter-process edge is connected to a remote peer. So the peer info attributes in the kernel should be
  // sufficient for route table.
  for (auto &send_actor : send_actors) {
    CNodePtr rpc_send_kernel = send_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_send_kernel);

    auto send_dst_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_send_kernel, kAttrSendDstRanks);
    auto send_dst_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_send_kernel, kAttrSendDstRoles);
    std::string send_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendSrcNodeName);
    std::string send_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendDstNodeName);
    std::string edge_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrInterProcessEdgeName);

    if (send_dst_ranks.empty() || send_dst_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of send node " << rpc_send_kernel->fullname_with_scope()
                        << " is invalid. send_dst_ranks: " << send_dst_ranks << ", send_dst_roles: " << send_dst_roles
                        << ", send_src_node_name: " << send_src_node_name
                        << ", send_dst_node_name: " << send_dst_node_name;
    }
    send_actor->set_inter_process_edge_name(edge_name);
    send_actor->SetRouteInfo(send_dst_ranks[0], send_dst_roles[0], send_src_node_name, send_dst_node_name);
  }
  for (auto &recv_actor : recv_actors) {
    CNodePtr rpc_recv_kernel = recv_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_recv_kernel);

    auto recv_src_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_recv_kernel, kAttrRecvSrcRanks);
    auto recv_src_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_recv_kernel, kAttrRecvSrcRoles);
    std::string recv_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvSrcNodeName);
    std::string recv_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvDstNodeName);
    std::string edge_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrInterProcessEdgeName);

    if (recv_src_ranks.empty() || recv_src_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of recv node " << rpc_recv_kernel->fullname_with_scope()
                        << " is invalid. recv_src_ranks: " << recv_src_ranks << ", recv_src_roles: " << recv_src_roles
                        << ", recv_src_node_name: " << recv_src_node_name
                        << ", recv_dst_node_name: " << recv_dst_node_name;
    }
    recv_actor->set_inter_process_edge_name(edge_name);
    recv_actor->SetRouteInfo(recv_src_ranks[0], recv_src_roles[0], recv_src_node_name, recv_dst_node_name);
  }
}

void RpcNodeScheduler::Schedule(const ActorSet *actor_set) {
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
}

ActorRouteTableProxyPtr RpcNodeScheduler::CreateRouteTableProxy() {
  ActorRouteTableProxyPtr actor_route_table_proxy;
  if (!ClusterContext::instance()->IsScheduler()) {
    auto node = ClusterContext::instance()->node();
    actor_route_table_proxy =
      std::make_shared<ActorRouteTableProxy>(std::dynamic_pointer_cast<ps::core::AbstractNode>(node));
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy);
  }
  return actor_route_table_proxy;
}
}  // namespace runtime
}  // namespace mindspore
