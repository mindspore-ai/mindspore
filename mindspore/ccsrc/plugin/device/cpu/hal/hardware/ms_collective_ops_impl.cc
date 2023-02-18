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

#include <numeric>
#include <functional>

#include "plugin/device/cpu/hal/hardware/ms_collective_ops_impl.h"
#include "utils/ms_context.h"
#include "actor/msg.h"
#include "plugin/device/cpu/hal/hardware/ms_collective_topo.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
namespace {
const char kCollectivePhaseRing[] = "ring";
const char kCollectivePhaseGather[] = "gather";
const char kCollectivePhaseReduce[] = "reduce";
const char kCollectivePhaseBroadcast[] = "broadcast";
}  // namespace

bool MSCollectiveOpsImpl::Initialize() {
  MS_EXCEPTION_IF_NULL(topo_node_);
  rank_id_ = SizeToUint(topo_node_->rank_id());
  return true;
}

template <typename T>
bool MSCollectiveOpsImpl::RingAllGather(const void *sendbuff, void *recvbuff, size_t send_count) {
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  size_t chunk_size = send_count;
  std::vector<size_t> chunk_sizes(rank_size_, chunk_size);

  // Store offsets to get every data chunk's address.
  std::vector<size_t> chunk_offset;
  for (size_t i = 0; i < rank_size_; i++) {
    size_t ofs = std::accumulate(chunk_sizes.begin(), chunk_sizes.begin() + SizeToLong(i), static_cast<size_t>(0),
                                 std::plus<size_t>());
    chunk_offset.push_back(ofs);
  }

  uint32_t send_to_rank = (rank_id_ + 1) % rank_size_;
  uint32_t recv_from_rank = (rank_id_ - 1 + rank_size_) % rank_size_;
  MS_LOG(DEBUG) << "Ring AllGather count:" << send_count << ", rank_size:" << rank_size_ << ", rank_id_:" << rank_id_
                << ", chunk_size:" << chunk_size << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", recv_from_rank:" << recv_from_rank;

  T *output_buff = reinterpret_cast<T *>(recvbuff);
  size_t src_size = send_count * sizeof(T);
  size_t dst_size = send_count * sizeof(T);
  int ret = memcpy_s(output_buff + chunk_offset[rank_id_], dst_size, sendbuff, src_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                  << ", dest size is " << dst_size << ", src size is " << src_size;
    return false;
  }
  return RingAllGatherImpl(send_to_rank, recv_from_rank, output_buff, chunk_offset, chunk_sizes);
}

template <typename T>
bool MSCollectiveOpsImpl::RingAllGatherImpl(uint32_t send_to_rank, uint32_t recv_from_rank, T *output_buff,
                                            const std::vector<size_t> &chunk_offset,
                                            const std::vector<size_t> &chunk_sizes) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // If enable recovery, set timeout 300s to prevent networking flapping.
  uint32_t timeout =
    context_ptr->get_param<bool>(MS_CTX_ENABLE_RECOVERY) ? kCollectiveCommMaxTimeout : kCollectiveCommTimeout;

  MS_EXCEPTION_IF_NULL(topo_node_);
  for (size_t i = 0; i < rank_size_ - 1; i++) {
    size_t send_chunk_index = (rank_id_ - i + rank_size_) % rank_size_;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];

    if (!topo_node_->SendAsync(send_to_rank, send_chunk, chunk_sizes[send_chunk_index] * sizeof(T))) {
      MS_LOG(ERROR) << "Failed to send data to rank: " << send_to_rank;
      return false;
    }

    size_t recv_chunk_index = (rank_id_ - i - 1 + rank_size_) % rank_size_;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << send_to_rank << ", recv_from_rank:" << recv_from_rank
                  << ", send count:" << chunk_sizes[send_chunk_index]
                  << ", recv count:" << chunk_sizes[recv_chunk_index] << ", iteration:" << i;

    MessageBase *message = nullptr;
    if (!topo_node_->Receive(recv_from_rank, &message, timeout)) {
      MS_LOG(ERROR) << "Failed to receive data from rank " << recv_from_rank;
      return false;
    }

    MS_EXCEPTION_IF_NULL(message);
    auto ret =
      memcpy_s(recv_chunk, chunk_sizes[recv_chunk_index] * sizeof(T), message->body.data(), message->body.length());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                    << ", dest size is " << (chunk_sizes[recv_chunk_index] * sizeof(T)) << ", src size is "
                    << message->body.length();
      return false;
    }
    delete message;
    message = nullptr;

    if (!topo_node_->WaitForSend(send_to_rank)) {
      MS_LOG(ERROR) << "Failed to send data to rank: " << send_to_rank;
      return false;
    }
  }
  return true;
}

template <typename T>
bool MSCollectiveOpsImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                    const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  MS_EXCEPTION_IF_NULL(topo_node_);
  rank_id_ = SizeToUint(topo_node_->rank_id());
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  if (rank_size_ == 1) {
    MS_LOG(INFO) << "Rank size is 1. Do nothing.";
    return true;
  }

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  if (group_to_global_ranks.empty()) {
    MS_LOG(ERROR) << "The group is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t global_root_rank = group_to_global_ranks[root];

  // Broadcast data to processes which are not the root.
  MS_LOG(DEBUG) << "Start broadcast from root to other processes.";
  if (rank_id_ == global_root_rank) {
    for (uint32_t i = 1; i < group_rank_size; i++) {
      uint32_t dst_rank = group_to_global_ranks[i];
      MS_LOG(DEBUG) << "Broadcast data to process " << dst_rank;

      if (!topo_node_->SendAsync(dst_rank, const_cast<void *>(sendbuff), count * sizeof(T)) ||
          !topo_node_->WaitForSend(dst_rank)) {
        MS_LOG(ERROR) << "Failed to send data to rank: " << dst_rank;
        return false;
      }
    }
  } else {
    MS_LOG(DEBUG) << "Broadcast receive from rank 0.";

    MessageBase *message = nullptr;
    if (!topo_node_->Receive(global_root_rank, &message)) {
      MS_LOG(ERROR) << "Failed to receive data from rank " << global_root_rank;
      return false;
    }

    MS_EXCEPTION_IF_NULL(message);
    int ret = memcpy_s(recvbuff, count * sizeof(T), message->body.data(), message->body.length());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                    << ", dest size is " << (count * sizeof(T)) << ", src size is " << message->body.length();
      return false;
    }
  }
  MS_LOG(DEBUG) << "End broadcast.";
  return true;
}

template <typename T>
bool MSCollectiveOpsImpl::AllGather(const void *sendbuff, void *recvbuff, size_t send_count) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  MS_EXCEPTION_IF_NULL(topo_node_);
  rank_id_ = SizeToUint(topo_node_->rank_id());
  rank_size_ = SizeToUint(topo_node_->rank_size());
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  if (rank_size_ == 1) {
    MS_LOG(INFO) << "Rank size is 1. Do nothing.";
    return true;
  }

  return RingAllGather<T>(sendbuff, recvbuff, send_count);
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
