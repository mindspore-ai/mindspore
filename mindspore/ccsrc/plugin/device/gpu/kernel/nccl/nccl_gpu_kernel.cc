/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nccl/nccl_gpu_kernel.h"
#include <memory>

namespace mindspore {
namespace kernel {
void NcclGpuKernelMod::SelectCollectiveHandle() {
  if (!LoadNvidiaCommLib()) {
    MS_LOG(EXCEPTION) << "Failed to load nivdia communication library.";
  }
}

bool NcclGpuKernelMod::LoadNvidiaCommLib() {
  std::string nvidia_comm_lib_name = "libnvidia_collective.so";
  auto loader = std::make_shared<device::CollectiveCommLibLoader>(nvidia_comm_lib_name);
  MS_EXCEPTION_IF_NULL(loader);
  if (!loader->Initialize()) {
    MS_LOG(EXCEPTION) << "Loading NCCL collective library failed.";
    return false;
  }
  nvidia_collective_handle_ = loader->collective_comm_lib_ptr();
  MS_EXCEPTION_IF_NULL(nvidia_collective_handle_);
  return true;
}

bool NcclGpuKernelMod::AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                 ncclRedOp_t reduce_op, cudaStream_t stream, const std::string &group_name) {
  auto allreduce_func = DlsymFuncObj(AllReduce, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                             allreduce_func(input_addr, output_addr, count, data_type, reduce_op, group_name, stream),
                             "ncclAllReduce failed");
  return true;
}

bool NcclGpuKernelMod::AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                 cudaStream_t stream, const std::string &group_name) {
  auto allgather_func = DlsymFuncObj(AllGather, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                             allgather_func(input_addr, output_addr, count, data_type, group_name, stream),
                             "ncclAllGather failed");
  return true;
}

bool NcclGpuKernelMod::ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                     ncclRedOp_t reduce_op, cudaStream_t stream, const std::string &group_name) {
  auto reducescatter_func = DlsymFuncObj(ReduceScatter, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(
    kernel_node_, reducescatter_func(input_addr, output_addr, count, data_type, reduce_op, group_name, stream),
    "ncclReduceScatter failed");
  return true;
}

bool NcclGpuKernelMod::Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                 int root, cudaStream_t stream, const std::string &group_name) {
  auto broadcast_func = DlsymFuncObj(Broadcast, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                             broadcast_func(input_addr, output_addr, count, data_type, root, group_name, stream),
                             "ncclBroadcast failed");
  return true;
}

bool NcclGpuKernelMod::Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank,
                            cudaStream_t stream, const std::string &group_name) {
  auto send_func = DlsymFuncObj(Send, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, send_func(send_addr, count, data_type, peer_rank, group_name, stream),
                             "ncclSend failed");
  return true;
}

bool NcclGpuKernelMod::Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                            const std::string &group_name) {
  auto recv_func = DlsymFuncObj(Recv, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, recv_func(recv_addr, count, data_type, peer_rank, group_name, stream),
                             "ncclRecv failed");
  return true;
}

bool NcclGpuKernelMod::GroupStart() {
  auto groupstart_func = DlsymFuncObj(GroupStart, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, groupstart_func(), "ncclGroupStart failed");
  return true;
}

bool NcclGpuKernelMod::GroupEnd() {
  auto groupend_func = DlsymFuncObj(GroupEnd, nvidia_collective_handle_);
  CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, groupend_func(), "ncclGroupEnd failed");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
