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

#include <functional>
#include <map>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include <string>

#include "include/backend/visible.h"
#include "include/backend/distributed/ps/constants.h"
#include "ps/core/communicator/communicator_base.h"
#include "ps/core/communicator/message.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/node.h"
#include "ps/core/node_info.h"
#include "ps/core/recovery_base.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ps {
namespace core {
class AbstractNode : public Node {
 public:
  AbstractNode()
      : heart_beat_thread_(nullptr),
        client_to_scheduler_thread_(nullptr),
        client_to_scheduler_(nullptr),
        client_to_server_(nullptr),
        server_(nullptr),
        server_thread_(nullptr),
        worker_num_(0),
        server_num_(0),
        is_connected_to_scheduler_(false),
        is_current_node_scale_in_(false),
        node_recovery_(nullptr),
        persistent_state_(PersistentState::NOT_ENABLE_PERSIST),
        scheduler_ip_(""),
        scheduler_port_(0),
        is_recover(false) {}
  ~AbstractNode() override;

  typedef void (AbstractNode::*ResponseHandler)(const std::shared_ptr<MessageMeta> &meta, const void *data,
                                                size_t size);
  typedef void (AbstractNode::*ServerHandler)(const std::shared_ptr<TcpConnection> &conn,
                                              const std::shared_ptr<MessageMeta> &meta, const Protos &protos,
                                              const void *data, size_t size);

  using VectorPtr = std::shared_ptr<std::vector<unsigned char>>;
  using RequestHandler = std::function<void(const std::shared_ptr<TcpConnection> &conn,
                                            const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size)>;
  using CancelSafeModeFn = std::function<void()>;

  bool Broadcast(const NodeRole &node_role, const std::string &message, int command,
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

  bool Send(const NodeRole &node_role, const uint32_t &rank_id, const void *message, size_t len, int command,
            VectorPtr *output = nullptr, const uint32_t &timeout = kCommTimeoutInSeconds);

  bool Send(const NodeRole &node_role, const uint32_t &rank_id, const std::string &msg, int command,
            VectorPtr *output = nullptr, const uint32_t &timeout = kCommTimeoutInSeconds);
  bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids, const std::vector<std::string> &msgs,
            int command, std::vector<VectorPtr> *output = nullptr, const uint32_t &timeout = kCommTimeoutInSeconds);

  // The interface that sends sync message to the scheduler.
  bool SendToScheduler(const void *message, size_t len, NodeCommand command, VectorPtr *output = nullptr,
                       const uint32_t &timeout = kCommTimeoutInSeconds);

  uint64_t CollectiveSendAsync(const NodeRole &node_role, const uint32_t &rank_id, const void *data, size_t size);

  using CheckFailReturnFun = std::function<bool()>;
  uint64_t FlCollectiveSendAsync(const CollectiveMessageMeta &collective_meta, const void *data, size_t size);
  bool FlCollectiveWait(const CollectiveMessageMeta &expect_meta, size_t expect_size, VectorPtr *output,
                        const uint32_t &timeout = kCommTimeoutInSeconds);

  std::pair<uint32_t, uint64_t> CollectiveReceiveAsync(const NodeRole &node_role, const uint32_t &rank_id,
                                                       VectorPtr *output);
  bool CollectiveWait(const std::pair<uint32_t, uint64_t> &request_id, const uint32_t &timeout = kCommTimeoutInSeconds);

  PersistentState persistent_state() const;
  void set_persistent_state(PersistentState persistent_state);

  uint32_t worker_num() const;
  uint32_t server_num() const;

  void set_worker_num(const uint32_t &worker_num);
  void set_server_num(const uint32_t &server_num);

  std::string scheduler_ip() const;
  void set_scheduler_ip(const std::string &scheduler_ip);

  uint16_t scheduler_port() const;
  void set_scheduler_port(const uint16_t &scheduler_port);

  ClusterState cluster_state() const;

  void set_handler(const RequestHandler &handler);
  void Response(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta, const void *data,
                size_t size);

  bool HasIterationFailed(uint32_t iteration_num) const;
  // register cancel SafeMode function to node
  void SetCancelSafeModeCallBack(const CancelSafeModeFn &fn) { cancelSafeModeFn_ = fn; }

  // server node and worker node send exception message to scheduler
  void SendFailMessageToScheduler(const std::string &node_role, const std::string &event_info);

 protected:
  virtual void Register(const std::shared_ptr<TcpClient> &client);
  bool Heartbeat(const std::shared_ptr<TcpClient> &client);
  void FetchServers(const std::shared_ptr<TcpClient> &client);

  void ProcessRegisterResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessHeartbeatResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);
  void ProcessFetchServersResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

  // Process the response messages about actor route table service.
  void ProcessReceiveSchedulerResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size);

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

  // The worker/server processes the scheduler recovery message from scheduelr
  void ProcessSchedulerRecovery(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                const Protos &, const void *data, size_t size);

  // The worker/server processes the SEND_EVENT message from scheduelr
  void ProcessEvent(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                    const Protos &protos, const void *data, size_t size);

  void StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client);
  void UpdateSchedulerTime();
  bool CheckSchedulerTimeout() const;
  bool Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout);
  bool WaitForDisconnect(const uint32_t &timeout);
  virtual bool InitClientToScheduler();
  void InitClientToServer();
  const std::shared_ptr<TcpClient> &GetOrCreateTcpClient(const uint32_t &rank_id,
                                                         const NodeRole &role = NodeRole::SERVER);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                       const uint32_t &timeout = kCommTimeoutInSeconds);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &, const void *, size_t size, const uint32_t &timeout = kCommTimeoutInSeconds);
  uint64_t SendCollectiveMeta(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
                              const Protos &protos, const void *data, size_t size);
  void ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                 const Protos &protos, const void *data, size_t size);
  void ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                       const Protos &protos, const void *data, size_t size);
  void NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta);
  void RunReceiveCallback(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                          size_t size);
  uint64_t NextExpectedRankRequestId(const uint32_t &rank_id);
  uint64_t NextActualRankRequestId(const uint32_t &rank_id);
  void InitCommandHandler();
  void RegisterActorRouteTableRspHandler();
  void InitServerHandler();

  // Register collective communication initialization response methods.
  virtual void RegisterInitCollectCommResphandler() {}

  // Register recovery response methods.
  virtual void RegisterRecoveryRespHandler() {}

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

  void UpdateClusterState(const ClusterState &state);

  void PersistMetaData();

  void ProcessPrepareBuildingNetwork(const std::shared_ptr<TcpConnection> &conn,
                                     const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                                     size_t size);

  bool FlCollectiveWaitInner(const CollectiveMessageMeta &expect_meta, VectorPtr *output, const uint32_t &timeout);
  void OnRecvCollectiveData(const MessageMeta &message_meta, const VectorPtr &data);
  void ConnectToScheduler();

  void ProcessScaleOutRollback(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                               const Protos &, const void *data, size_t size);

  std::unique_ptr<std::thread> heart_beat_thread_;
  std::unique_ptr<std::thread> client_to_scheduler_thread_;
  std::shared_ptr<TcpClient> client_to_scheduler_;
  std::shared_ptr<TcpClient> client_to_server_;
  // the key is: <node_role,rank_id>, the value is: <ip, port>
  std::map<std::pair<NodeRole, uint32_t>, std::pair<std::string, uint16_t>> nodes_address_;
  // the map's key is: rank_id
  std::map<std::pair<NodeRole, uint32_t>, std::shared_ptr<TcpClient>> connected_nodes_;

  // the key is <rank_id, rank_request_id>
  std::map<std::pair<uint32_t, uint64_t>, VectorPtr> received_data_;
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

  // send_rank_id, recv CollectiveMessageMeta and data
  std::unordered_map<uint32_t, std::vector<std::pair<CollectiveMessageMeta, std::shared_ptr<std::vector<uint8_t>>>>>
    fl_received_data_;
  std::mutex fl_receive_mutex_;
  std::condition_variable fl_receive_cond_;

  // Workers and servers launch the server to process command: FINISH,SCALE_OUT,SCALE_IN,SEND_METADATA
  std::shared_ptr<TcpServer> server_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<std::thread> message_callback_thread_;

  uint32_t worker_num_;
  uint32_t server_num_;
  std::atomic<bool> is_connected_to_scheduler_;
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

  // Recovery for worker/server node.
  std::unique_ptr<RecoveryBase> node_recovery_;

  // The state of the persistent storage, such as ready to be persisted, in the process of being persisted, has
  // completed the persistence, etc.
  std::atomic<PersistentState> persistent_state_;

  // The ip of scheduler.
  std::string scheduler_ip_;
  // The port of scheduler.
  uint16_t scheduler_port_;

  // Synchronize all node metadata from the scheduler.
  std::unordered_map<std::string, NodeInfo> all_nodes_info_;
  RequestHandler request_handler_;

  std::unordered_map<std::string, std::shared_ptr<CommunicatorBase>> communicators_;
  std::mutex communicator_mutex_;
  std::mutex cluster_state_mutex_;

  size_t failed_iteration_num_ = 0;
  bool iteration_failed_ = false;
  CancelSafeModeFn cancelSafeModeFn_;

  std::atomic<bool> is_recover;
};
using AbstractNodePtr = std::shared_ptr<AbstractNode>;
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
