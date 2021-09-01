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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_COMMUNICATOR_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_COMMUNICATOR_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "proto/ps.pb.h"
#include "ps/core/server_node.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/cluster_config.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/communicator/communicator_base.h"
#include "ps/core/communicator/tcp_msg_handler.h"
#include "ps/core/comm_util.h"
#include "ps/constants.h"

namespace mindspore {
namespace ps {
namespace core {
const std::unordered_map<TcpUserCommand, std::string> kUserCommandToMsgType = {
  {TcpUserCommand::kPush, "push"},
  {TcpUserCommand::kPull, "pull"},
  {TcpUserCommand::kCount, "count"},
  {TcpUserCommand::kReachThreshold, "countReachThreshold"},
  {TcpUserCommand::kResetCount, "resetCnt"},
  {TcpUserCommand::kGetMetadata, "getMetadata"},
  {TcpUserCommand::kUpdateMetadata, "updateMetadata"},
  {TcpUserCommand::kCounterEvent, "counterEvent"},
  {TcpUserCommand::kPullWeight, "pullWeight"},
  {TcpUserCommand::kPushWeight, "pushWeight"},
  {TcpUserCommand::kSyncIteration, "syncIteration"},
  {TcpUserCommand::kNotifyLeaderToNextIter, "notifyLeaderToNextIter"},
  {TcpUserCommand::kPrepareForNextIter, "prepareForNextIter"},
  {TcpUserCommand::kProceedToNextIter, "proceedToNextIter"},
  {TcpUserCommand::kEndLastIter, "endLastIter"},
  {TcpUserCommand::kStartFLJob, "startFLJob"},
  {TcpUserCommand::kUpdateModel, "updateModel"},
  {TcpUserCommand::kGetModel, "getModel"},
  {TcpUserCommand::kPushMetrics, "pushMetrics"},
  {TcpUserCommand::kNewInstance, "newInstance"},
  {TcpUserCommand::kQueryInstance, "queryInstance"},
  {TcpUserCommand::kEnableFLS, "enableFLS"},
  {TcpUserCommand::kDisableFLS, "disableFLS"}};

class TcpCommunicator : public CommunicatorBase {
 public:
  explicit TcpCommunicator(const std::shared_ptr<TaskExecutor> &task_executor, AbstractNode *node)
      : task_executor_(task_executor),
        server_num_(0),
        worker_num_(0),
        scheduler_ip_(""),
        scheduler_port_(0),
        abstrace_node_(node) {}
  ~TcpCommunicator() = default;

  bool Start() override;
  bool Stop() override;

  void RegisterMsgCallBack(const std::string &msg_type, const MessageCallback &cb) override;
  void RegisterEventCallback(const core::ClusterEvent &event, const EventCallback &event_cb);

  template <class T>
  bool SendPbRequest(const T &pb_msg, const uint32_t &rank_id, TcpUserCommand command,
                     std::shared_ptr<std::vector<unsigned char>> *output = nullptr) {
    const std::string &msg_str = pb_msg.SerializeAsString();
    std::shared_ptr<unsigned char[]> msg(new unsigned char[msg_str.size()]);
    MS_ERROR_IF_NULL_W_RET_VAL(msg, false);
    size_t dest_size = msg_str.size();
    size_t src_size = msg_str.size();
    if (memcpy_s(msg.get(), dest_size, msg_str.c_str(), src_size) != EOK) {
      MS_LOG(EXCEPTION) << "Memcpy_s error";
    }

    if (output != nullptr) {
      if (!abstrace_node_->Send(NodeRole::SERVER, rank_id, msg, msg_str.size(), static_cast<int>(command), output)) {
        MS_LOG(ERROR) << "Sending protobuffer message to server " << rank_id << " failed.";
        return false;
      }
    } else {
      if (!abstrace_node_->Send(NodeRole::SERVER, rank_id, msg, msg_str.size(), static_cast<int>(command))) {
        MS_LOG(ERROR) << "Sending protobuffer message to server " << rank_id << " failed.";
        return false;
      }
    }
    return true;
  }

 private:
  std::shared_ptr<TaskExecutor> task_executor_;

  TcpMsgCallback tcp_msg_callback_;
  OnNodeEventCallback event_callback_;

  uint32_t server_num_;
  uint32_t worker_num_;

  std::string scheduler_ip_;
  uint16_t scheduler_port_;

  AbstractNode *abstrace_node_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_COMMUNICATOR_H_
