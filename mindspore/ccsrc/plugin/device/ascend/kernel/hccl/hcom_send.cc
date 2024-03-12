/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_send.h"

#include <string>
#include <utility>

#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "distributed/cluster/actor_route_table_proxy.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/common/utils/parallel_context.h"
#include "proto/topology.pb.h"
#include "runtime/graph_scheduler/actor/rpc/rpc_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"

namespace mindspore {
namespace kernel {
constexpr int64_t DYNAMIC_SHAPE = -1;
HcomSendKernel::~HcomSendKernel() {
  if (client_) {
    try {
      (void)client_->Disconnect(server_url_);
      client_->Finalize();
    } catch (const std::exception &) {
      MS_LOG(ERROR) << "Failed to disconnect and finalize for rpc client";
    }
    client_ = nullptr;
  }
}

int HcomSendKernel::SendShapeForDynamic() {
  // send shape by rpc
  if (client_ == nullptr) {
    // rpc 1. create client and init
    client_ = std::make_unique<mindspore::distributed::rpc::TCPClient>();
    MS_EXCEPTION_IF_NULL(client_);

    if (!client_->Initialize()) {
      MS_LOG(EXCEPTION) << "Failed to initialize rpc server for send actor.";
    }

    // rpc 2. lookuproute
    uint32_t src_rank = LongToUint(mindspore::parallel::ParallelContext::GetInstance()->global_rank());  // src rank id
    std::string server_url_key = std::to_string(src_rank) + "_" + std::to_string(dest_rank_) + "_tag_" +
                                 std::to_string(op_tag) + "_rpc_addr";  // rpc addr
    op_tag++;

    auto node = distributed::cluster::ClusterContext::instance()->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(node);
    MS_EXCEPTION_IF_NULL(cgn);
    auto actor_route_table_proxy = std::make_shared<mindspore::distributed::cluster::ActorRouteTableProxy>(cgn);
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy);
    auto peer_actor_address = actor_route_table_proxy->LookupRoute(server_url_key);
    server_url_ = peer_actor_address.ip() + ":" + std::to_string(peer_actor_address.port());

    // rpc 3. connect
    size_t retry_count = 60;
    if (!client_->Connect(server_url_, retry_count)) {
      MS_LOG(EXCEPTION) << "Failed to connect to server of actor, server_url: " << server_url_;
    }
    MS_LOG(DEBUG) << "server key is " << server_url_key << ", server url is " << server_url_;
  }

  // rpc 4. send shape
  std::vector<int64_t> real_shape = hccl_kernel_output_shape_list_[0];
  std::string msg_str;  // convert shape to string
  for (size_t i = 0; i < real_shape.size(); ++i) {
    msg_str += std::to_string(real_shape[i]);
    if (i < real_shape.size() - 1) {
      msg_str += "_";
    }
  }
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_EXCEPTION_IF_NULL(message);
  message->to = AID("", server_url_);
  message->body = msg_str;  // use body to send

  MS_LOG(DEBUG) << "send msg is " << msg_str;

  client_->SendAsync(std::move(message));

  if (!client_->Flush(server_url_)) {
    MS_LOG(EXCEPTION) << "Failed to flush client for server " << server_url_;
  }
  return KRET_OK;
}

int HcomSendKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }

  // update output_size_list_
  output_size_list_.clear();

  for (ulong i = 0; i < loop_size_; ++i) {
    size_t size = 0;
    if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
    }
    output_size_list_.push_back(size);
  }

  // if dynamic shape, send the shape
  if (!get_shape_attr_flag_) {
    MS_EXCEPTION_IF_NULL(primitive_);
    auto value = primitive_->GetAttr("shape");
    MS_EXCEPTION_IF_NULL(value);

    if (!value->isa<ValueSequence>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The shape is not value sequence: " << value->ToString();
    }

    std::vector<ValuePtr> vals = value->cast<ValueSequencePtr>()->value();
    if (vals.empty()) {
      return KRET_OK;
    }

    if (vals[0]->isa<Int32Imm>()) {
      auto shape_v = GetValue<std::vector<int32_t>>(value);
      is_dynamic_shape_ = (std::count(shape_v.cbegin(), shape_v.cend(), DYNAMIC_SHAPE) > 0);
    } else if (vals[0]->isa<Int64Imm>()) {
      auto shape_v = GetValue<std::vector<int64_t>>(value);
      is_dynamic_shape_ = (std::count(shape_v.cbegin(), shape_v.cend(), DYNAMIC_SHAPE) > 0);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "The value of shape is not int32 or int64: " << value->ToString();
    }
    get_shape_attr_flag_ = true;
  }

  if (!is_dynamic_shape_) {
    return KRET_OK;
  }

  SendShapeForDynamic();
  return KRET_OK;
}

bool HcomSendKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                            const std::vector<KernelTensor *> &, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcomSend launch";
  if (inputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid HcomSend input size or data type size (" << inputs.size() << ", "
                  << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclSend(inputs[0]->device_ptr(), hccl_count_,
                                                               hccl_data_type_list_[0], dest_rank_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomSend failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
