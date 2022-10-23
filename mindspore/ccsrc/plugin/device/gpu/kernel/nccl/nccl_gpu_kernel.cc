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
  use_mpi_ = false;
  if (use_mpi_) {
    collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
    MS_EXCEPTION_IF_NULL(collective_handle_);
  } else {
    if (!LoadNvidiaCommLib()) {
      MS_LOG(EXCEPTION) << "Failed to load nivdia communication library.";
    }
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
  if (use_mpi_) {
    auto all_reduce_funcptr =
      reinterpret_cast<kernel::AllReduce>(dlsym(const_cast<void *>(collective_handle_), "AllReduce"));
    MS_EXCEPTION_IF_NULL(all_reduce_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, (*all_reduce_funcptr)(input_addr, output_addr, count, data_type, reduce_op, stream, group_name),
      "ncclAllReduce failed");
  } else {
    auto allreduce_func = DlsymFuncObj(AllReduce, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               allreduce_func(input_addr, output_addr, count, data_type, reduce_op, group_name, stream),
                               "ncclAllReduce failed");
  }
  return true;
}

bool NcclGpuKernelMod::AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                 cudaStream_t stream, const std::string &group_name) {
  if (use_mpi_) {
    auto all_gather_funcptr =
      reinterpret_cast<kernel::AllGather>(dlsym(const_cast<void *>(collective_handle_), "AllGather"));
    MS_EXCEPTION_IF_NULL(all_gather_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               (*all_gather_funcptr)(input_addr, output_addr, count, data_type, stream, group_name),
                               "ncclAllGather failed");
  } else {
    auto allgather_func = DlsymFuncObj(AllGather, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               allgather_func(input_addr, output_addr, count, data_type, group_name, stream),
                               "ncclAllGather failed");
  }
  return true;
}

bool NcclGpuKernelMod::ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                     ncclRedOp_t reduce_op, cudaStream_t stream, const std::string &group_name) {
  if (use_mpi_) {
    auto reduce_scatter_funcptr =
      reinterpret_cast<kernel::ReduceScatter>(dlsym(const_cast<void *>(collective_handle_), "ReduceScatter"));
    MS_EXCEPTION_IF_NULL(reduce_scatter_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, (*reduce_scatter_funcptr)(input_addr, output_addr, count, data_type, reduce_op, stream, group_name),
      "ncclReduceScatter failed");
  } else {
    auto reducescatter_func = DlsymFuncObj(ReduceScatter, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, reducescatter_func(input_addr, output_addr, count, data_type, reduce_op, group_name, stream),
      "ncclReduceScatter failed");
  }
  return true;
}

bool NcclGpuKernelMod::Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                 int root, cudaStream_t stream, const std::string &group_name) {
  if (use_mpi_) {
    auto broadcast_funcptr =
      reinterpret_cast<kernel::Broadcast>(dlsym(const_cast<void *>(collective_handle_), "Broadcast"));
    MS_EXCEPTION_IF_NULL(broadcast_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, (*broadcast_funcptr)(input_addr, output_addr, count, data_type, root, stream, group_name),
      "ncclBroadcast failed");
  } else {
    auto broadcast_func = DlsymFuncObj(Broadcast, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               broadcast_func(input_addr, output_addr, count, data_type, root, group_name, stream),
                               "ncclBroadcast failed");
  }
  return true;
}

bool NcclGpuKernelMod::Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank,
                            cudaStream_t stream, const std::string &group_name) {
  if (use_mpi_) {
    auto nccl_send_func = reinterpret_cast<kernel::Send>(dlsym(const_cast<void *>(collective_handle_), "Send"));
    MS_EXCEPTION_IF_NULL(nccl_send_func);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, (*nccl_send_func)(send_addr, count, data_type, peer_rank, stream, group_name), "ncclSend failed");
  } else {
    auto send_func = DlsymFuncObj(Send, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, send_func(send_addr, count, data_type, peer_rank, group_name, stream),
                               "ncclSend failed");
  }
  return true;
}

bool NcclGpuKernelMod::Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                            const std::string &group_name) {
  if (use_mpi_) {
    auto nccl_recv_func = reinterpret_cast<kernel::Recv>(dlsym(const_cast<void *>(collective_handle_), "Recv"));
    MS_EXCEPTION_IF_NULL(nccl_recv_func);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_, (*nccl_recv_func)(recv_addr, count, data_type, peer_rank, stream, group_name), "ncclRecv failed");
  } else {
    auto recv_func = DlsymFuncObj(Recv, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, recv_func(recv_addr, count, data_type, peer_rank, group_name, stream),
                               "ncclRecv failed");
  }
  return true;
}

bool NcclGpuKernelMod::GroupStart() {
  if (use_mpi_) {
    auto nccl_gstart_func =
      reinterpret_cast<kernel::GroupStart>(dlsym(const_cast<void *>(collective_handle_), "GroupStart"));
    MS_EXCEPTION_IF_NULL(nccl_gstart_func);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, (*nccl_gstart_func)(), "ncclGroupStart failed");
  } else {
    auto groupstart_func = DlsymFuncObj(GroupStart, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, groupstart_func(), "ncclGroupStart failed");
  }
  return true;
}

bool NcclGpuKernelMod::GroupEnd() {
  if (use_mpi_) {
    auto nccl_gend_func = reinterpret_cast<kernel::GroupEnd>(dlsym(const_cast<void *>(collective_handle_), "GroupEnd"));
    MS_EXCEPTION_IF_NULL(nccl_gend_func);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, (*nccl_gend_func)(), "ncclGroupEnd failed");
  } else {
    auto groupend_func = DlsymFuncObj(GroupEnd, nvidia_collective_handle_);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_, groupend_func(), "ncclGroupEnd failed");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
