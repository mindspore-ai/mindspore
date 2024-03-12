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

#include "plugin/device/ascend/kernel/hccl/hcom_receive.h"

#include <mutex>
#include <string>
#include <functional>
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "distributed/cluster/actor_route_table_proxy.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/common/utils/parallel_context.h"
#include "proto/topology.pb.h"
#include "kernel/framework_utils.h"
#include "plugin/device/cpu/kernel/rpc/rpc_recv_kernel.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/distributed/rpc/tcp/constants.h"

namespace mindspore {
namespace kernel {
HcomReceiveKernel::~HcomReceiveKernel() {
  if (server_) {
    try {
      server_->Finalize();
    } catch (const std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize for tcp server";
    }
    server_ = nullptr;
  }
}

int HcomReceiveKernel::ReceiveShapeForDynamic() {
  // recv shape by rpc
  if (server_ == nullptr) {
    // rpc 1. create server, one server for one recv op
    server_ = std::make_unique<mindspore::distributed::rpc::TCPServer>(
      false, distributed::cluster::ClusterContext::instance()->port_range());
    MS_EXCEPTION_IF_NULL(server_);

    // rpc 2. Initialize
    if (!server_->Initialize()) {
      MS_LOG(EXCEPTION) << "Failed to initialize rpc server for recv actor";
    }

    // rpc 3. SetMessageHandler, the handler is a callback and is triggered in new thread
    server_->SetMessageHandler(std::bind(&HcomReceiveKernel::HandleMessage, this, std::placeholders::_1));

    // rpc 4. register route
    std::string ip = server_->GetIP();
    uint32_t port = server_->GetPort();
    std::string server_url = ip + ":" + std::to_string(port);

    uint32_t dst_rank = LongToUint(mindspore::parallel::ParallelContext::GetInstance()->global_rank());  // dst rank id

    // Register the server address to route table.
    // The key of route is src_rank + "_" + dst_rank + "_tag_" + op_tag + "_rpc_addr"
    // The op_tag is used to identify multiple send/recv ops
    uint32_t kRemoteFuncId = 0;
    std::string inter_process_edge_name = std::to_string(src_rank_) + "_" + std::to_string(dst_rank) + "_tag_" +
                                          std::to_string(op_tag) + "_rpc_addr";  // rpc addr
    op_tag++;
    MS_LOG(DEBUG) << "Start server for recv actor. Server address: " << server_url
                  << ", remote function id: " << kRemoteFuncId
                  << ", inter-process edge name: " << inter_process_edge_name;
    distributed::cluster::topology::ActorAddress recv_actor_addresss;
    recv_actor_addresss.set_actor_id(inter_process_edge_name);
    recv_actor_addresss.set_ip(ip);
    recv_actor_addresss.set_port(port);
    recv_actor_addresss.set_func_id(kRemoteFuncId);

    auto node = distributed::cluster::ClusterContext::instance()->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(node);
    MS_EXCEPTION_IF_NULL(cgn);
    auto actor_route_table_proxy = std::make_shared<mindspore::distributed::cluster::ActorRouteTableProxy>(cgn);
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy);
    if (!actor_route_table_proxy->RegisterRoute(inter_process_edge_name, recv_actor_addresss)) {
      MS_LOG(EXCEPTION) << "Failed to register route for " << inter_process_edge_name << " " << server_url
                        << " when starting server.";
    }
  }

  // rpc 5. handle message
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock);

  MS_LOG(DEBUG) << "handle msg done, the real shape is " << real_shape_;
  return KRET_OK;
}

int HcomReceiveKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }

  // update output_size_list_
  output_size_list_.clear();

  if (!get_shape_attr_flag_) {
    auto shape_v = GetValue<std::vector<int64_t>>(primitive_->GetAttr("shape"));
    is_dynamic_shape_ = (std::count(shape_v.cbegin(), shape_v.cend(), -1) > 0);
    get_shape_attr_flag_ = true;
  }

  if (!is_dynamic_shape_) {
    for (size_t i = 0; i < loop_size_; ++i) {
      size_t size = 0;
      if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), hccl_kernel_output_shape_list_[i], &size)) {
        MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
      }
      output_size_list_.push_back(size);
    }

    return KRET_OK;
  }

  real_shape_.clear();
  ReceiveShapeForDynamic();
  // resize
  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), real_shape_, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  output_size_list_.push_back(size);

  // update hccl_count_
  if (!HcomUtil::GetHcomCount(primitive_, hccl_data_type_list_, {real_shape_}, inputs.size(), &hccl_count_)) {
    MS_LOG(ERROR) << "GetHcomCount fail!";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

MessageBase *HcomReceiveKernel::HandleMessage(MessageBase *const msg) {
  if (msg == nullptr) {
    return distributed::rpc::NULL_MSG;
  }

  // The message content is in the msg->body, which is a string, and the format is "a_b_c", if shape is [a, b, c]
  std::vector<int64_t> recv_shape;
  std::string token;
  std::istringstream t_string(msg->body);
  while (std::getline(t_string, token, '_')) {
    recv_shape.push_back(std::stoll(token));
  }

  std::lock_guard<std::mutex> lock(mtx);
  real_shape_ = recv_shape;

  cv.notify_all();
  return distributed::rpc::NULL_MSG;
}

void HcomReceiveKernel::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  if (!is_dynamic_shape_) {
    return;
  }
  outputs[kIndex0]->SetShapeVector(real_shape_);
  size_t all_size = LongToSize(std::accumulate(real_shape_.begin(), real_shape_.end(), 1, std::multiplies<int64_t>()));
  outputs[kIndex0]->set_size(all_size * UnitSizeInBytes(outputs[kIndex0]->dtype_id()));
}

bool HcomReceiveKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcomReceive launch";
  if (outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid HcomReceive outputs size or data type size (" << outputs.size() << ", "
                  << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclRecv(outputs[0]->device_ptr(), hccl_count_,
                                                               hccl_data_type_list_[0], src_rank_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomReceive failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
