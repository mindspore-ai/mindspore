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

#ifndef MINDSPORE_CCSRC_PS_CORE_SCHEDULER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_SCHEDULER_NODE_H_

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>

#include "ps/core/cluster_config.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/tcp_client.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/node_manager.h"
#include "ps/core/node.h"
#include "ps/core/communicator/request_process_result_code.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/constants.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/communicator/http_server.h"
#include "ps/core/leader_scaler.h"
#include "ps/core/recovery_base.h"
#include "ps/core/instance_manager.h"

namespace mindspore {
namespace ps {
namespace core {
class SchedulerNode : public Node {
 public:
  SchedulerNode()
      : server_(nullptr),
        scheduler_thread_(nullptr),
        update_state_thread_(nullptr),
        restful_thread_(nullptr),
        http_server_(nullptr),
        client_thread_(nullptr),
        is_client_started_(false),
        leader_scaler_(nullptr),
        scheduler_recovery_(nullptr) {}
  ~SchedulerNode() override;

  typedef void (SchedulerNode::*ResponseHandler)(const std::shared_ptr<TcpServer> &server,
                                                 const std::shared_ptr<TcpConnection> &conn,
                                                 const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                 size_t size);

  bool Start(const uint32_t &timeout = PSContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;
  bool Finish(const uint32_t &timeout = kTimeoutInSeconds) override;

 private:
  void Initialize();

  void InitCommandHandler();
  void CreateTcpServer();
  void StartUpdateClusterStateTimer();
  const std::shared_ptr<TcpClient> &GetOrCreateClient(const NodeInfo &node_info);

  void ProcessHeartbeat(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessRegister(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                       const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFinish(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                     const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFetchMetadata(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process scale_out_done messages from workers/servers
  void ProcessScaleOutDone(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                           const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  // Process scale_in_done messages from workers/servers
  void ProcessScaleInDone(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                          const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process scale_in_done messages from workers/servers
  void ProcessSendEvent(const std::shared_ptr<TcpServer> &server, const std::shared_ptr<TcpConnection> &conn,
                        const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // After scheduler collects all registered message, it actively sends finish to the node connected by the client.
  void SendMetadata(const std::shared_ptr<TcpClient> &client, uint32_t rank_id);
  // After scheduler collects all finish message, it actively sends finish to the node connected by the client.
  void SendFinish(const std::shared_ptr<TcpClient> &client);

  // After scheduler collects all scale_out_done message, it actively sends scale_out_done to the node connected by the
  // client.
  void SendScaleOutDone(const std::shared_ptr<TcpClient> &client);

  // After scheduler collects all scale_in_done message, it actively sends scale_out_done to the node connected by the
  // client.
  void SendScaleInDone(const std::shared_ptr<TcpClient> &client);
  // After scheduler receive SEND_EVENT message, it will broadcast the event to all other nodes.
  void SendEvent(const std::shared_ptr<TcpClient> &client, const uint32_t &event);

  // Handle the scale out http request, then delegate to the leader scaler to process scale out asynchronously.
  void ProcessScaleOut(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the scale in http request, then delegate to the leader scaler to process scale in asynchronously.
  void ProcessScaleIn(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the get nodes info http request Synchronously.
  void ProcessGetNodesInfo(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the get cluster state http request Synchronously.
  void ProcessGetClusterState(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the new instance http request Synchronously.
  void ProcessNewInstance(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the query instance http request Synchronously.
  void ProcessQueryInstance(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the enable FLS http request Synchronously.
  void ProcessEnableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the disable FLS http request Synchronously.
  void ProcessDisableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  // check whether the cluster is in the ready state.
  RequestProcessResult CheckIfClusterReady();

  // check whether the node id is legal.
  RequestProcessResult CheckIfNodeIdLegal(const std::vector<std::string> &node_ids);

  void StartRestfulServer(const std::string &address, std::uint16_t port, size_t thread_num = 10);
  void StopRestfulServer();

  std::shared_ptr<TcpServer> server_;
  std::unique_ptr<std::thread> scheduler_thread_;
  std::unique_ptr<std::thread> update_state_thread_;
  std::unordered_map<NodeCommand, ResponseHandler> handlers_;

  NodeManager node_manager_;

  // In this thread will start a http server.
  std::unique_ptr<std::thread> restful_thread_;
  std::shared_ptr<HttpServer> http_server_;

  std::unordered_map<std::string, std::shared_ptr<TcpClient>> connected_nodes_;

  std::unique_ptr<std::thread> client_thread_;
  std::atomic<bool> is_client_started_;

  std::unique_ptr<LeaderScaler> leader_scaler_;

  std::unordered_map<std::string, OnRequestReceive> callbacks_;

  // Used to persist and obtain metadata information for scheduler.
  std::unique_ptr<RecoveryBase> scheduler_recovery_;

  // The node id of scale in nodes.
  std::vector<std::string> scale_in_node_ids_;

  std::unique_ptr<InstanceManager> instance_manager_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_SCHEDULER_NODE_H_
