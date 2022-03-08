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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_RECV_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_RECV_ACTOR_H_

#include <set>
#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <condition_variable>
#include "runtime/graph_scheduler/actor/rpc/rpc_actor.h"

namespace mindspore {
namespace runtime {
// RecvActor inherits from RpcActor and it's used to receive data from other processes.
class RecvActor : public RpcActor {
 public:
  explicit RecvActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                     const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                     GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                     const std::set<size_t> &modifiable_ref_output_indexes)
      : RpcActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                 modifiable_ref_input_indexes, modifiable_ref_output_indexes, KernelTransformType::kRecvActor) {}
  ~RecvActor() override = default;

  // Besides set the op context, this method also notify the message handler to 'RunOpInterProcessData'.
  void SetOpcontext(OpContext<DeviceTensor> *const op_context) override;

  // This method means the op context is invalid now. If the message handler is called while the op context is invalid,
  // it should be blocked until 'SetOpcontext' is called.
  void ResetOpcontext() override;

  // Set recv actor's source peer info, in another word, recv actor's input.
  void SetRouteInfo(uint32_t src_rank, const std::string &src_role, const std::string &recv_src_node_name,
                    const std::string &recv_dst_node_name) override;

  // Start recv actor server and register this server address to actor route table in scheduler by proxy.
  bool StartServer();

 protected:
  // When an inter-process data received, this method is called.
  void RunOpInterProcessData(const std::shared_ptr<MessageBase> &msg, OpContext<DeviceTensor> *const context);

  // Besides the checking method in base class AbstractActor, condition of inter-process arrows should be checked for
  // recv actor.
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const override;

 private:
  // The message callback of the tcp server.
  void HandleMessage(const std::shared_ptr<MessageBase> &msg);

  // The network address of this recv actor. It's generated automatically by rpc module.
  std::string ip_;
  uint32_t port_;

  std::unique_ptr<TCPServer> server_;

  // The variables used to ensure thread-safe of op context visited by recv actor.
  bool is_context_valid_;
  std::mutex context_mtx_;
  std::condition_variable context_cv_;
};

using RecvActorPtr = std::shared_ptr<RecvActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_RECV_ACTOR_H_
