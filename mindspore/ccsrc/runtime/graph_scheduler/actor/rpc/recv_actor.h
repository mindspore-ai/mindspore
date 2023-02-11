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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace runtime {
using CPUDeviceAddress = device::cpu::CPUDeviceAddress;
// RecvActor inherits from RpcActor and it's used to receive data from other processes.
class RecvActor : public RpcActor {
 public:
  explicit RecvActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                     const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                     GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                     const std::set<size_t> &modifiable_ref_output_indexes)
      : RpcActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                 modifiable_ref_input_indexes, modifiable_ref_output_indexes, KernelTransformType::kRecvActor),
        server_(nullptr),
        is_context_valid_(false),
        recv_data_(nullptr),
        ip_(""),
        port_(0) {}
  ~RecvActor() override;

  // Besides set the op context, this method also notify the message handler to 'RunOpInterProcessData'.
  void SetOpcontext(OpContext<DeviceTensor> *const op_context) override;

  // This method means the op context is invalid now. If the message handler is called while the op context is invalid,
  // it should be blocked until 'SetOpcontext' is called.
  void ResetOpcontext() override;

  // Update the context status after loop_count_actor is launched.
  void UpdateStatus() override;

  // Set recv actor's source peer info, in another word, recv actor's input.
  void SetRouteInfo(uint32_t src_rank, const std::string &src_role, const std::string &recv_src_node_name,
                    const std::string &recv_dst_node_name) override;

  // Start recv actor server and register this server address to actor route table in scheduler by proxy.
  bool StartServer();

  // Finalize rpc server.
  void Clear() override;

  void StopRpcAtException() override;

 protected:
  // Besides the checking method in base class AbstractActor, condition of inter-process arrows should be checked for
  // recv actor.
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const override;

  // When an inter-process data received, this method is called.
  void RunOpInterProcessData(MessageBase *const msg, OpContext<DeviceTensor> *const context);

  // Besides erasing input data and input controls when finish actor running, inter-process inputs should be erased.
  void EraseInput(const OpContext<DeviceTensor> *context) override;

  // Before calling the Run method in KernelActor, some preprocess like inferring shape should be done. So rewrite the
  // Run method.
  void Run(OpContext<DeviceTensor> *const context) override;

  // Set the message handler of the server.
  virtual void SetMessageHandler();

  // Parse finalize command message from received message.
  virtual void ParseFinalizeReqData(size_t data_len, const MessageBase *const msg, bool *need_finalize) {}

  /**
   * @description: The callback set to rpc module to allocate message(Raw pointer).
   * @param {size_t} size: The message size.
   * @return {void *}: A pointer to the newly allocated memory.
   */
  virtual void *AllocateMessage(size_t size);

  /**
   * @description: Allocate memory by DeviceResManager.
   * @param {size_t} size: memory buffer's size.
   * @return {void *}
   */
  void *AllocateMemByDeviceRes(size_t size);

  std::unique_ptr<RPCServerBase> server_;

  // The variables used to ensure thread-safe of op context visited by recv actor.
  bool is_context_valid_;
  std::mutex context_mtx_;
  std::condition_variable context_cv_;

  // The received data which should be allocated by framework.
  // It will be used for copying the buffer from the kernel function.
  std::shared_ptr<CPUDeviceAddress> recv_data_;

 private:
  // Create abstract and add to the abstract list.
  void AddArgSpecForInput(AbstractBasePtrList *args_spec_list, const ShapeVector &shapes, TypeId data_type,
                          size_t input_index) const;

  // Parse the protobuf message from the given buffer. The format is as below.
  // |--------22 bytes------|---4 bytes--|PB data size bytes| data size bytes |
  // |RPC_DYNAMIC_SHAPE_DATA|PB data size|      PB data     | real data       |
  // Return dynamic shape data length.
  size_t ParseDynamicShapeData(const RpcDataPtr &dynamic_shape_data, size_t data_size,
                               AbstractBasePtrList *args_spec_list, size_t count);

  // After Recv actor receives data from a remote peer, the data could be with dynamic shape so we need to preprocess
  // it, e.g., infer shape for RpcRecv kernel and call Resize().
  void PreprocessRemoteInput(const MessageBase *const msg, bool *need_finalize);

  // The message callback of the rpc server.
  MessageBase *HandleMessage(MessageBase *const msg);

  // The network address of this recv actor. It's generated automatically by rpc module.
  std::string ip_;
  uint32_t port_;
};

using RecvActorPtr = std::shared_ptr<RecvActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_RPC_RECV_ACTOR_H_
