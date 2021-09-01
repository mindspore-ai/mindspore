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

#ifndef MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_

#include <utility>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <unordered_map>

#include "ps/core/node.h"
#include "ps/core/communicator/message.h"
#include "ps/core/follower_scaler.h"
#include "utils/ms_exception.h"
#include "ps/constants.h"
#include "ps/core/node_info.h"
#include "ps/core/recovery_base.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/communicator/communicator_base.h"

namespace mindspore {
namespace ps {
namespace core {
class FollowerScaler;
class AbstractNode : public Node {
 public:
  AbstractNode()
      : heart_beat_thread_(nullptr),
        client_to_scheduler_thread_(nullptr),
        client_to_scheduler_(nullptr),
        server_(nullptr),
        server_thread_(nullptr),
        worker_num_(-1),
        server_num_(-1),
        is_current_node_scale_in_(false),
        follower_scaler_(nullptr),
        node_recovery_(nullptr),
        scheduler_ip_(""),
        scheduler_port_(0) {}
  ~AbstractNode() override = default;

  typedef void (AbstractNode::*ResponseHandler)(const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                size_t size);
  typedef void (AbstractNode::*ServerHandler)(const std::shared_ptr<TcpConnection> &conn,
                                              const std::shared_ptr<MessageMeta> &meta, const Protos &protos,
                                              const void *data, size_t size);

  using DataPtr = std::shared_ptr<unsigned char[]>;
  using VectorPtr = std::shared_ptr<std::vector<unsigned char>>;
  using RequestHandler =
    std::function<void(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const DataPtr &data, size_t size)>;

  bool Broadcast(const NodeRole &node_role, const DataPtr &message, size_t size, int command,
                 const uint32_t &timeout = kCommTimeoutInSeconds);

  // When the business layer finish scale out, it should call this function
  void set_ready_for_scale_out();
  // When the business layer finish scale in, it should call this function
  void set_ready_for_scale_in();

  // Send scale_out_done instructions to the scheduler.
  void set_scale_out_done();

  // Send scale_in_done instructions to the scheduler.
  void set_scale_in_done();

  // The worker/server sends the event to the scheduler, and then the scheduler broadcasts this event to all nodes.
  void BroadcastEvent(const uint32_t &event);

  // Set the callback corresponding to the event.
  void RegisterEventCallback(const ClusterEvent &event, const EventCallback &event_cb);
  // Set the callback corresponding to the custom event.
  void RegisterCustomEventCallback(const uint32_t &event, const EventCallback &event_cb);

  bool Send(const NodeRole &node_role, const uint32_t &rank_id, const DataPtr &data, size_t len, int command,
            const uint32_t &timeout = kTimeoutInSeconds);
  bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids, const std::vector<DataPtr> &data,
            const std::vector<size_t> &lens, int command, const uint32_t &timeout = kTimeoutInSeconds);
  bool Send(const NodeRole &node_role, const uint32_t &rank_id, const DataPtr &message, size_t len, int command,
            VectorPtr *output, const uint32_t &timeout = kTimeoutInSeconds);
  bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids, const std::vector<DataPtr> &data,
            const std::vector<size_t> &data_lens, int command, std::vector<VectorPtr> *output,
            const uint32_t &timeout = kTimeoutInSeconds);

  uint64_t CollectiveSendAsync(const NodeRole &node_role, const uint32_t &rank_id, const void *data, size_t size);
  std::pair<uint32_t, uint64_t> CollectiveReceiveAsync(const NodeRole &node_role, const uint32_t &rank_id,
                                                       VectorPtr *output);
  bool CollectiveWait(const std::pair<uint32_t, uint64_t> &request_id, const uint32_t &timeout = kCommTimeoutInSeconds);

  // Initialize the scaler for server to process before/after scaling operations.
  bool InitFollowerScaler();

  // Register barriers before scaling operations for server.
  void RegisterFollowerScalerBarrierBeforeScaleOut(const std::string &module, const BarrierBeforeScaleOut &barrier);
  void RegisterFollowerScalerBarrierBeforeScaleIn(const std::string &module, const BarrierBeforeScaleIn &barrier);

  // Register handlers after scaling operations for server.
  void RegisterFollowerScalerHandlerAfterScaleOut(const std::string &module, const HandlerAfterScaleOut &handler);
  void RegisterFollowerScalerHandlerAfterScaleIn(const std::string &module, const HandlerAfterScaleIn &handler);

  int32_t worker_num() const;
  int32_t server_num() const;

  void set_worker_num(const int32_t &worker_num);
  void set_server_num(const int32_t &server_num);

  std::string scheduler_ip() const;
  void set_scheduler_ip(const std::string &scheduler_ip);

  uint16_t scheduler_port() const;
  void set_scheduler_port(const uint16_t &scheduler_port);

  ClusterState cluster_state() const;

  void set_handler(const RequestHandler &handler);
  void Response(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta, const void *data,
                size_t size);

  std::shared_ptr<CommunicatorBase> GetOrCreateHttpComm(const std::string &ip, uint16_t port,
                                                        const std::shared_ptr<TaskExecutor> &task_executor);
  std::shared_ptr<CommunicatorBase> GetOrCreateTcpComm(const std::string &scheduler_ip, std::int16_t scheduler_port,
                                                       uint32_t worker_num, uint32_t server_num,
                                                       const std::shared_ptr<TaskExecutor> &task_executor);

 protected:
  void Register(const std::shared_ptr<TcpClient> &client);
  bool Heartbeat(const std::shared_ptr<TcpClient> &client);
  void FetchServers(const std::shared_ptr<TcpClient> &client);

  void ProcessRegisterResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessHeartbeatResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFetchServersResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  void ProcessSendMetadata(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                           const Protos &protos, const void *data, size_t size);
  void ProcessFinish(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                     const Protos &protos, const void *data, size_t size);

  void ProcessScaleOut(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &protos, const void *data, size_t size);

  void ProcessScaleIn(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                      const Protos &protos, const void *data, size_t size);

  // The worker/server processes the scale_out_done message from scheduelr
  void ProcessScaleOutDone(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                           const Protos &protos, const void *data, size_t size);
  // The worker/server processes the scale_in_done message from scheduelr
  void ProcessScaleInDone(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                          const Protos &protos, const void *data, size_t size);

  // The worker/server processes the SEND_EVENT message from scheduelr
  void ProcessEvent(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                    const Protos &protos, const void *data, size_t size);

  void StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client);
  void UpdateSchedulerTime();
  bool CheckSchedulerTimeout() const;
  bool Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout);
  bool WaitForDisconnect(const uint32_t &timeout);
  bool InitClientToScheduler();
  const std::shared_ptr<TcpClient> &GetOrCreateTcpClient(const uint32_t &rank_id);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                       const uint32_t &timeout = kCommTimeoutInSeconds);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &, const void *, size_t size, const uint32_t &timeout = kCommTimeoutInSeconds);
  uint64_t SendMessageAsync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                            const Protos &protos, const void *data, size_t size);
  void ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                 const void *data, size_t size);
  void ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &protos, const void *data, size_t size);
  void NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta);
  void RunReceiveCallback(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                          size_t size);
  uint64_t NextExpectedRankRequestId(const uint32_t &rank_id);
  uint64_t NextActualRankRequestId(const uint32_t &rank_id);
  void InitCommandHandler();
  void InitServerHandler();

  // when initializing the node, should initializing the node info.
  void InitNodeInfo(const NodeRole &role);
  // Initialize worker num and server num by cluster config.
  void InitNodeNum();
  // Node recover by cluster config.
  bool Recover();

  // Trigger the callback corresponding to the event.
  void OnEventCallback(const ClusterEvent &event);
  // Trigger the callback corresponding to the custom event.
  void OnCustomEventCallback(const uint32_t &event);

  bool IsWorkerOrServer0(const std::unordered_map<std::string, NodeInfo> &info);

  void CreateTcpServer();

  std::unique_ptr<std::thread> heart_beat_thread_;
  std::unique_ptr<std::thread> client_to_scheduler_thread_;
  std::shared_ptr<TcpClient> client_to_scheduler_;

  // the key is: <node_role,rank_id>, the value is: <ip, port>
  std::map<std::pair<NodeRole, uint32_t>, std::pair<std::string, uint16_t>> nodes_address_;
  // the map's key is: rank_id
  std::unordered_map<uint32_t, std::shared_ptr<TcpClient>> connected_nodes_;

  // the key is <rank_id, rank_request_id>
  std::map<std::pair<uint32_t, uint64_t>, std::shared_ptr<std::vector<unsigned char>>> received_data_;
  std::mutex receive_callbacks_mutex_;
  // the key is <rank_id, rank_request_id>
  std::map<std::pair<uint32_t, uint64_t>, MessageCallback> receive_callbacks_;
  std::condition_variable receive_cond_;

  // the key is rank_id, the value is rank_id's expected request_id
  std::unordered_map<uint32_t, uint64_t> expected_rank_request_ids_;
  // the key is rank_id, the value is rank_id's actual request_id
  std::unordered_map<uint32_t, uint64_t> actual_rank_request_ids_;
  std::mutex rank_request_ids_mutex;
  timeval scheduler_time_{0, 0};
  std::unordered_map<NodeCommand, ResponseHandler> handlers_;
  std::unordered_map<NodeCommand, ServerHandler> server_handler_;

  // Workers and servers launch the server to process command: FINISH,SCALE_OUT,SCALE_IN,SEND_METADATA
  std::shared_ptr<TcpServer> server_;
  std::unique_ptr<std::thread> server_thread_;

  int32_t worker_num_;
  int32_t server_num_;

  // Identify whether the current node is a scale in node.
  std::atomic<bool> is_current_node_scale_in_;

  // Each ClusterEvent corresponds to a EventCallback to process the event.
  std::map<ClusterEvent, EventCallback> event_to_callback_;

  // Each custom event corresponds to a EventCallback to process the event.
  // This event is sent to the scheduler, and then the scheduler broadcasts this event to all nodes.
  // for example:
  // In order to ensure the consistency of the cluster, the server broadcasts an iteration_end event to notify all other
  // nodes to modify the iteration status
  std::map<uint32_t, EventCallback> custom_event_to_callback_;

  // Scaler for worker/server node.
  std::unique_ptr<FollowerScaler> follower_scaler_;

  // Recovery for worker/server node.
  std::unique_ptr<RecoveryBase> node_recovery_;

  // The ip of scheduler.
  std::string scheduler_ip_;
  // The port of scheduler.
  uint16_t scheduler_port_;

  // Synchronize all node metadata from the scheduler.
  std::unordered_map<std::string, NodeInfo> all_nodes_info_;
  RequestHandler request_handler_;

  std::unordered_map<std::string, std::shared_ptr<CommunicatorBase>> communicators_;
  std::mutex communicator_mutex_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
