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
void RpcNodeScheduler::Initialize() {
  rpc_actor_set_ = std::make_shared<RpcActorSet>();
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
}

RpcActorSetPtr RpcNodeScheduler::Build(const GraphCompilerInfo &) {
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
  std::vector<RpcActorPtr> rpc_actors;
  (void)rpc_actors.insert(rpc_actors.end(), rpc_actor_set_->send_actors_.begin(), rpc_actor_set_->send_actors_.end());
  (void)rpc_actors.insert(rpc_actors.end(), rpc_actor_set_->recv_actors_.begin(), rpc_actor_set_->recv_actors_.end());

  // Create route table proxy for each rpc actor and set.
  for (auto &rpc_actor : rpc_actors) {
    auto proxy = CreateRouteTableProxy();
    MS_EXCEPTION_IF_NULL(proxy);
    rpc_actor->SetActorRouteRableProxy(proxy);
  }

  return rpc_actor_set_;
}

void RpcNodeScheduler::Link(const ActorSet *) {
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
  std::vector<SendActorPtr> send_actors = rpc_actor_set_->send_actors_;
  std::vector<RecvActorPtr> recv_actors = rpc_actor_set_->recv_actors_;
  // The inter-process edge is connected to a remote peer. So the peer info attributes in the kernel should be
  // sufficient for route table.
  for (auto &send_actor : send_actors) {
    CNodePtr rpc_send_kernel = send_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_send_kernel);

    auto send_dst_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_send_kernel, kAttrSendDstRanks);
    auto send_dst_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_send_kernel, kAttrSendDstRoles);
    std::string send_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendSrcNodeName);
    std::string send_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_send_kernel, kAttrSendDstNodeName);

    if (send_dst_ranks.empty() || send_dst_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of send node " << rpc_send_kernel->fullname_with_scope()
                        << " is invalid. send_dst_ranks: " << send_dst_ranks << ", send_dst_roles: " << send_dst_roles
                        << ", send_src_node_name: " << send_src_node_name
                        << ", send_dst_node_name: " << send_dst_node_name;
    }
    send_actor->SetInterProcessEdgeName(send_src_node_name, send_dst_node_name);
    send_actor->SetRouteInfo(send_dst_ranks[0], send_dst_roles[0], send_src_node_name, send_dst_node_name);
  }
  for (auto &recv_actor : recv_actors) {
    CNodePtr rpc_recv_kernel = recv_actor->kernel();
    MS_EXCEPTION_IF_NULL(rpc_recv_kernel);

    auto recv_src_ranks = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(rpc_recv_kernel, kAttrRecvSrcRanks);
    auto recv_src_roles = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(rpc_recv_kernel, kAttrRecvSrcRoles);
    std::string recv_src_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvSrcNodeName);
    std::string recv_dst_node_name = common::AnfAlgo::GetNodeAttr<std::string>(rpc_recv_kernel, kAttrRecvDstNodeName);

    if (recv_src_ranks.empty() || recv_src_roles.empty()) {
      MS_LOG(EXCEPTION) << "The attributes of recv node " << rpc_recv_kernel->fullname_with_scope()
                        << " is invalid. recv_src_ranks: " << recv_src_ranks << ", recv_src_roles: " << recv_src_roles
                        << ", recv_src_node_name: " << recv_src_node_name
                        << ", recv_dst_node_name: " << recv_dst_node_name;
    }
    recv_actor->SetInterProcessEdgeName(recv_src_node_name, recv_dst_node_name);
    recv_actor->SetRouteInfo(recv_src_ranks[0], recv_src_roles[0], recv_src_node_name, recv_dst_node_name);
  }
}

void RpcNodeScheduler::Schedule() {
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
  // Must start server and register route table before looking up route and connecting.

  // Start servers of recv actors and register route table.
  for (auto &recv_actor : rpc_actor_set_->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    if (!recv_actor->StartServer()) {
      MS_LOG(EXCEPTION) << "Failed to start server for the recv actor.";
    }
  }
  // Lookup route and connect to servers for send actors.
  for (auto &send_actor : rpc_actor_set_->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    if (!send_actor->ConnectServer()) {
      MS_LOG(EXCEPTION) << "Failed to connect servers for the send actor.";
    }
  }
}

void RpcNodeScheduler::InsertSendActor(const SendActorPtr &send_actor) {
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
  MS_EXCEPTION_IF_NULL(send_actor);
  (void)rpc_actor_set_->send_actors_.emplace_back(send_actor);
}

void RpcNodeScheduler::InsertRecvActor(const RecvActorPtr &recv_actor) {
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);
  MS_EXCEPTION_IF_NULL(recv_actor);
  (void)rpc_actor_set_->recv_actors_.emplace_back(recv_actor);
}

void RpcNodeScheduler::SetOpcontext(OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(rpc_actor_set_);

  for (auto &recv_actor : rpc_actor_set_->recv_actors_) {
    MS_EXCEPTION_IF_NULL(recv_actor);
    recv_actor->SetOpcontext(op_context);
  }
  for (auto &send_actor : rpc_actor_set_->send_actors_) {
    MS_EXCEPTION_IF_NULL(send_actor);
    send_actor->SetOpcontext(op_context);
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
