/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "include/cluster/rpc/collective_ops_impl.h"
#include <complex>
#include "utils/ms_context.h"

namespace mindspore {
namespace fl {
namespace server {
namespace {
using complex64 = std::complex<float>;
const char kCollectivePhaseRing[] = "ring";
const char kCollectivePhaseGather[] = "gather";
const char kCollectivePhaseReduce[] = "reduce";
const char kCollectivePhaseBroadcast[] = "broadcast";

template <typename T>
void calculate(T *output_buff, T *tmp_buff, size_t count, CollectiveOpReduceType reduce_op) {
  if (reduce_op == CollectiveOpReduceType::Reduce_Sum) {
    for (size_t j = 0; j < count; j++) {
      output_buff[j] += tmp_buff[j];
    }
  }
  if (reduce_op == CollectiveOpReduceType::Reduce_Max) {
    for (size_t j = 0; j < count; j++) {
      output_buff[j] = (tmp_buff[j] > output_buff[j]) ? tmp_buff[j] : output_buff[j];
    }
  }
  if (reduce_op == CollectiveOpReduceType::Reduce_Min) {
    for (size_t j = 0; j < count; j++) {
      output_buff[j] = (tmp_buff[j] < output_buff[j]) ? tmp_buff[j] : output_buff[j];
    }
  }
  if (reduce_op == CollectiveOpReduceType::Reduce_Prod) {
    for (size_t j = 0; j < count; j++) {
      output_buff[j] = tmp_buff[j] * output_buff[j];
    }
  }
}

}  // namespace

template <typename T>
bool CollectiveOpsImpl::RingAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                      CollectiveOpReduceType reduce_op, const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];

  if (recvbuff != sendbuff) {
    size_t src_size = count * sizeof(T);
    size_t dst_size = count * sizeof(T);
    auto ret = Memcpy(recvbuff, dst_size, sendbuff, src_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")";
      return false;
    }
  }

  uint32_t rank_size = group_rank_size;
  size_t chunk_size = count / rank_size;
  size_t remainder_size = count % rank_size;
  std::vector<size_t> chunk_sizes(rank_size, chunk_size);
  // The rest of the data should be assigned to each chunk.
  for (size_t i = 0; i < remainder_size; i++) {
    chunk_sizes[i]++;
  }
  // Store offsets to get every data chunk's address.
  std::vector<size_t> chunk_offset;
  for (size_t i = 0; i < rank_size; i++) {
    size_t ofs =
      std::accumulate(chunk_sizes.begin(), chunk_sizes.begin() + i, static_cast<size_t>(0), std::plus<size_t>());
    chunk_offset.push_back(ofs);
  }

  T *output_buff = reinterpret_cast<T *>(recvbuff);
  uint32_t send_to_rank = (group_rank + 1) % rank_size;
  uint32_t recv_from_rank = (group_rank - 1 + rank_size) % rank_size;
  MS_LOG(DEBUG) << "AllReduce count:" << count << ", rank_size:" << rank_size << ", rank_id_:" << group_rank
                << ", chunk_size:" << chunk_size << ", remainder_size:" << remainder_size
                << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", recv_from_rank:" << recv_from_rank;

  return RunRingAllReduce<T>(send_to_rank, recv_from_rank, chunk_sizes, chunk_offset, output_buff, reduce_op,
                             group_info);
}

// Implementation of RingAllReduce.
template <typename T>
bool CollectiveOpsImpl::RunRingAllReduce(uint32_t send_to_rank, uint32_t recv_from_rank,
                                         const std::vector<size_t> &chunk_sizes,
                                         const std::vector<size_t> &chunk_offset, T *output_buff,
                                         CollectiveOpReduceType reduce_op, const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output_buff, false);
  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];

  // Ring ReduceScatter.
  MS_LOG(DEBUG) << "Start Ring ReduceScatter.";
  uint32_t rank_size = group_rank_size;
  for (size_t i = 0; i < rank_size - 1; i++) {
    // Step 1: Async send data to next rank.
    size_t send_chunk_index = (group_rank - i + rank_size) % rank_size;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_chunk_count = chunk_sizes[send_chunk_index];

    auto send_req_id = node_->CollectiveSendAsync(node_role_, group_to_global_ranks[send_to_rank], send_chunk,
                                                  send_chunk_count * sizeof(T));

    // Step 2: Async receive data to next rank and wait until it's done.
    size_t recv_chunk_index = (group_rank - i - 1 + rank_size) % rank_size;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    auto recv_chunk_count = chunk_sizes[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring ReduceScatter send_to_rank:" << group_to_global_ranks[send_to_rank]
                  << ", recv_from_rank:" << group_to_global_ranks[recv_from_rank]
                  << ", send chunk index:" << send_chunk_index << ", send count:" << send_chunk_count
                  << ", recv chunk index:" << recv_chunk_index << ", recv count:" << recv_chunk_count
                  << ", for index:" << i;

    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = node_->CollectiveReceiveAsync(node_role_, group_to_global_ranks[recv_from_rank], &rec_ptr);
    if (!node_->CollectiveWait(rec_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "Ring ReduceScatter wait receiving [" << rec_req_id.first << "," << rec_req_id.second
                    << "] failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(rec_ptr);
    auto tmp_recv_chunk = reinterpret_cast<T *>(rec_ptr->data());
    // Step 3: Reduce the data so we can overlap the time cost of send.
    calculate(recv_chunk, tmp_recv_chunk, recv_chunk_count, reduce_op);

    // Step 4: Wait until send is done.
    if (!node_->Wait(send_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "Ring ReduceScatter wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring ReduceScatter.";
  // Ring AllGather.
  MS_LOG(DEBUG) << "Start Ring AllGather.";
  for (size_t i = 0; i < rank_size - 1; i++) {
    size_t send_chunk_index = (group_rank - i + 1 + rank_size) % rank_size;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_chunk_count = chunk_sizes[send_chunk_index];
    auto send_req_id = node_->CollectiveSendAsync(node_role_, group_to_global_ranks[send_to_rank], send_chunk,
                                                  send_chunk_count * sizeof(T));

    size_t recv_chunk_index = (group_rank - i + rank_size) % rank_size;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    auto recv_chunk_count = chunk_sizes[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << group_to_global_ranks[send_to_rank]
                  << ", recv_from_rank:" << group_to_global_ranks[recv_from_rank]
                  << ", send chunk index:" << send_chunk_index << ", send count:" << send_chunk_count
                  << ", recv chunk index:" << recv_chunk_index << ", recv count:" << recv_chunk_count
                  << ", for index:" << i;

    auto expect_size = recv_chunk_count * sizeof(T);
    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = node_->CollectiveReceiveAsync(node_role_, group_to_global_ranks[recv_from_rank], &rec_ptr);
    if (!node_->CollectiveWait(rec_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "Ring AllGather wait receiving " << rec_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(rec_ptr);

    auto ret = Memcpy(recv_chunk, expect_size, rec_ptr->data(), rec_ptr->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << recv_chunk_count * sizeof(T) << ", src size is " << rec_ptr->size();
      return false;
    }
    if (!node_->Wait(send_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "RingAllReduce wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring AllGather.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::ReduceBroadcastAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                                 CollectiveOpReduceType reduce_op,
                                                 const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];

  uint32_t rank_size = group_rank_size;
  MS_LOG(DEBUG) << "Reduce Broadcast AllReduce rank_size:" << rank_size << ", rank_id_:" << group_rank
                << ", count:" << count;

  size_t src_size = count * sizeof(T);
  size_t dst_size = count * sizeof(T);
  int ret = Memcpy(recvbuff, dst_size, sendbuff, src_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                  << ", dest size is " << dst_size << ", src size is " << src_size;
    return false;
  }
  T *output_buff = reinterpret_cast<T *>(recvbuff);
  // Reduce data to rank 0 process.

  MS_LOG(DEBUG) << "Start Reduce to rank 0 process.";
  if (group_rank == 0) {
    for (uint32_t i = 1; i < rank_size; i++) {
      std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;

      auto rec_req_id = node_->CollectiveReceiveAsync(node_role_, group_to_global_ranks[i], &rec_ptr);
      if (!node_->CollectiveWait(rec_req_id, comm_op_timeout_)) {
        MS_LOG(ERROR) << "Reduce wait receiving " << rec_req_id << " failed.";
        return false;
      }
      MS_EXCEPTION_IF_NULL(rec_ptr);
      auto tmp_recv_chunk = reinterpret_cast<T *>(rec_ptr->data());  // recv_str size has checked in FlCollectiveWait
      calculate(output_buff, tmp_recv_chunk, count, reduce_op);
    }
  } else {
    MS_LOG(DEBUG) << "Reduce send data to rank 0 process.";
    auto send_req_id = node_->CollectiveSendAsync(node_role_, group_to_global_ranks[0], sendbuff, count * sizeof(T));
    if (!node_->Wait(send_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "Reduce wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Reduce.";

  // Broadcast data to not 0 rank process.
  MS_LOG(DEBUG) << "Start broadcast from rank 0 to other processes.";
  if (group_rank == 0) {
    for (uint32_t i = 1; i < rank_size; i++) {
      MS_LOG(DEBUG) << "Broadcast data to process " << i;
      auto send_req_id =
        node_->CollectiveSendAsync(node_role_, group_to_global_ranks[i], output_buff, count * sizeof(T));
      if (!node_->Wait(send_req_id, comm_op_timeout_)) {
        MS_LOG(ERROR) << "Broadcast wait sending " << send_req_id << " failed.";
        return false;
      }
    }
  } else {
    MS_LOG(DEBUG) << "Broadcast receive from rank 0.";
    auto expect_size = count * sizeof(T);
    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = node_->CollectiveReceiveAsync(node_role_, group_to_global_ranks[0], &rec_ptr);
    if (!node_->CollectiveWait(rec_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "Broadcast wait receiving " << rec_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(rec_ptr);
    ret = Memcpy(output_buff, expect_size, rec_ptr->data(), rec_ptr->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << expect_size << ", src size is " << rec_ptr->size();
      return false;
    }
  }
  MS_LOG(DEBUG) << "End broadcast.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::RingAllGather(const void *sendbuff, void *recvbuff, size_t send_count,
                                      const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  size_t chunk_size = send_count;
  std::vector<size_t> chunk_sizes(rank_size_, chunk_size);

  if (rank_size_ == 1) {
    int ret = Memcpy(recvbuff, chunk_size * sizeof(T), sendbuff, chunk_size * sizeof(T));
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << (chunk_size * sizeof(T));
      return false;
    }
    return true;
  }

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];
  // Store offsets to get every data chunk's address.
  std::vector<size_t> chunk_offset;
  for (size_t i = 0; i < group_rank_size; i++) {
    size_t ofs = std::accumulate(chunk_sizes.begin(), chunk_sizes.begin() + SizeToLong(i), static_cast<size_t>(0),
                                 std::plus<size_t>());
    chunk_offset.push_back(ofs);
  }

  uint32_t send_to_rank = (group_rank + 1) % group_rank_size;
  uint32_t recv_from_rank = (group_rank - 1 + group_rank_size) % group_rank_size;
  MS_LOG(DEBUG) << "Ring AllGather count:" << send_count << ", group_rank_size:" << group_rank_size
                << ", rank_id_:" << rank_id_ << ", group_rank:" << group_rank << ", chunk_size:" << chunk_size
                << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", recv_from_rank:" << recv_from_rank;

  T *output_buff = reinterpret_cast<T *>(recvbuff);
  size_t src_size = send_count * sizeof(T);
  size_t dst_size = send_count * sizeof(T);
  int ret = Memcpy(output_buff + chunk_offset[group_rank], dst_size, sendbuff, src_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                  << ", dest size is " << dst_size << ", src size is " << src_size;
    return false;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // If enable recovery, set timeout 300s to prevent networking flapping.
  // Ring AllGather.
  for (size_t i = 0; i < group_rank_size - 1; i++) {
    size_t send_chunk_index = (group_rank - i + group_rank_size) % group_rank_size;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_req_id = node_->CollectiveSendAsync(node_role_, group_to_global_ranks[send_to_rank], send_chunk,
                                                  chunk_sizes[send_chunk_index] * sizeof(T));
    size_t recv_chunk_index = (group_rank - i - 1 + group_rank_size) % group_rank_size;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << group_to_global_ranks[send_to_rank]
                  << ", recv_from_rank:" << group_to_global_ranks[recv_from_rank]
                  << ", send count:" << chunk_sizes[send_chunk_index]
                  << ", recv count:" << chunk_sizes[recv_chunk_index] << ", iteration:" << i;

    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = node_->CollectiveReceiveAsync(node_role_, group_to_global_ranks[recv_from_rank], &recv_str);
    if (!node_->CollectiveWait(recv_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(recv_str);
    ret = Memcpy(recv_chunk, chunk_sizes[recv_chunk_index] * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << (chunk_sizes[recv_chunk_index] * sizeof(T)) << ", src size is "
                    << recv_str->size();
      return false;
    }
    if (!node_->Wait(send_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring AllGather.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                  const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

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
    for (uint32_t i = 0; i < group_rank_size; i++) {
      if (i == root) {
        int ret = Memcpy(recvbuff, count * sizeof(T), sendbuff, count * sizeof(T));
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                        << ", broadcast size is " << (count * sizeof(T));
          return false;
        }
      } else {
        uint32_t dst_rank = group_to_global_ranks[i];
        MS_LOG(DEBUG) << "Broadcast data to process " << dst_rank;
        auto send_req_id = node_->CollectiveSendAsync(node_role_, dst_rank, sendbuff, count * sizeof(T));
        if (!node_->Wait(send_req_id, comm_op_timeout_)) {
          MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
          return false;
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "Broadcast receive from rank " << global_root_rank;
    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = node_->CollectiveReceiveAsync(node_role_, global_root_rank, &recv_str);
    if (!node_->CollectiveWait(recv_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(recv_str);
    int ret = Memcpy(recvbuff, count * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << (count * sizeof(T)) << ", src size is " << recv_str->size();
      return false;
    }
  }
  MS_LOG(DEBUG) << "End broadcast.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::Send(const void *sendbuff, size_t count, uint32_t root,
                             const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  if (group_to_global_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t global_root_rank = group_to_global_ranks[root];

  // Gather data to processes which are not the root.
  MS_LOG(DEBUG) << "Send data to process " << global_root_rank;
  auto send_req_id = node_->CollectiveSendAsync(node_role_, global_root_rank, sendbuff, count * sizeof(T));
  if (!node_->Wait(send_req_id, comm_op_timeout_)) {
    MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
    return false;
  }
  MS_LOG(DEBUG) << "End Send.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::Recv(void *recvbuff, size_t count, uint32_t root, const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  if (group_to_global_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t global_root_rank = group_to_global_ranks[root];

  // Gather data to processes which are not the root.
  MS_LOG(DEBUG) << "Start Recv from root from " << global_root_rank;
  std::shared_ptr<std::vector<unsigned char>> recv_str;
  auto recv_req_id = node_->CollectiveReceiveAsync(node_role_, global_root_rank, &recv_str);
  if (!node_->CollectiveWait(recv_req_id, comm_op_timeout_)) {
    MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(recv_str);
  int ret = Memcpy(recvbuff, count * sizeof(T), recv_str->data(), recv_str->size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                  << ", dest size is " << (count * sizeof(T)) << ", src size is " << recv_str->size();
    return false;
  }
  MS_LOG(DEBUG) << "End Recv.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::Gather(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                               const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];
  uint32_t global_root_rank = group_to_global_ranks[root];

  // Gather data to processes which are not the root.
  MS_LOG(DEBUG) << "Start Gather from root to other processes.";
  if (rank_id_ == global_root_rank) {
    T *output_buff = reinterpret_cast<T *>(recvbuff);
    for (uint32_t i = 0; i < group_rank_size; i++) {
      T *recv_chunk = output_buff + i * count;
      if (i == group_rank) {
        int ret = Memcpy(recv_chunk, count * sizeof(T), sendbuff, count * sizeof(T));
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                        << ", Gather size is " << (count * sizeof(T));
          return false;
        }
      } else {
        MS_LOG(DEBUG) << "Gather receive from rank 0.";
        uint32_t dst_rank = group_to_global_ranks[i];
        std::shared_ptr<std::vector<unsigned char>> recv_str;
        auto recv_req_id = node_->CollectiveReceiveAsync(node_role_, dst_rank, &recv_str);
        if (!node_->CollectiveWait(recv_req_id, comm_op_timeout_)) {
          MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
          return false;
        }
        MS_EXCEPTION_IF_NULL(recv_str);
        int ret = Memcpy(recv_chunk, count * sizeof(T), recv_str->data(), recv_str->size());
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                        << ", dest size is " << (count * sizeof(T)) << ", src size is " << recv_str->size();
          return false;
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "Gather data to process " << global_root_rank;
    auto send_req_id = node_->CollectiveSendAsync(node_role_, global_root_rank, sendbuff, count * sizeof(T));
    if (!node_->Wait(send_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Gather.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::Scatter(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                const CommunicationGroupInfo &group_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  auto group_to_global_ranks = group_info.group_to_global_ranks;
  auto global_to_group_ranks = group_info.global_to_group_ranks;
  if (group_to_global_ranks.empty() || global_to_group_ranks.empty()) {
    MS_LOG(ERROR) << "The group ranks is empty.";
    return false;
  }
  uint32_t group_rank_size = SizeToUint(group_info.group_ranks.size());
  uint32_t group_rank = global_to_group_ranks[rank_id_];
  uint32_t global_root_rank = group_to_global_ranks[root];

  // Broadcast data to processes which are not the root.
  MS_LOG(DEBUG) << "Start Scatter from root to other processes.";
  if (rank_id_ == global_root_rank) {
    T *in_buff = static_cast<T *>(const_cast<void *>(sendbuff));
    for (size_t i = 0; i < group_rank_size; i++) {
      T *send_chunk = in_buff + i * count;
      if (i == group_rank) {
        int ret = Memcpy(recvbuff, count * sizeof(T), send_chunk, count * sizeof(T));
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                        << ", Scatter size is " << (count * sizeof(T));
          return false;
        }
      } else {
        uint32_t dst_rank = group_to_global_ranks[i];
        MS_LOG(DEBUG) << "Scatter data to process " << dst_rank << ",count is " << count;
        auto send_req_id = node_->CollectiveSendAsync(node_role_, dst_rank, send_chunk, count * sizeof(T));
        if (!node_->Wait(send_req_id, comm_op_timeout_)) {
          MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
          return false;
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "Scatter receive from rank " << global_root_rank;
    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = node_->CollectiveReceiveAsync(node_role_, global_root_rank, &recv_str);
    if (!node_->CollectiveWait(recv_req_id, comm_op_timeout_)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(recv_str);
    int ret = Memcpy(recvbuff, count * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy error, errorno(" << ret << ")"
                    << ", dest size is " << (count * sizeof(T)) << ", src size is " << recv_str->size();
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Scatter.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                  const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info) {
  // The collective communication API does not support calling Send and Recv concurrently with multiple threads;
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);

  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  switch (node_role_) {
    case ps::core::WORKER:
      rank_size_ = group_info.size;
      break;
    default:
      MS_LOG(ERROR) << "The node role " << node_role_ << " for collective communication is invalid.";
      return false;
  }

  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }

  if (count >= rank_size_) {
    return RingAllReduce<T>(sendbuff, recvbuff, count, (CollectiveOpReduceType)reduce_op, group_info);
  } else {
    return ReduceBroadcastAllReduce<T>(sendbuff, recvbuff, count, (CollectiveOpReduceType)reduce_op, group_info);
  }
}

template <typename T>
bool CollectiveOpsImpl::AllGather(const void *sendbuff, void *recvbuff, size_t send_count,
                                  const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  switch (node_role_) {
    case ps::core::WORKER:
      rank_size_ = group_info.size;
      break;
    default:
      MS_LOG(ERROR) << "The node role " << node_role_ << " for collective communication is invalid.";
      return false;
  }
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return RingAllGather<T>(sendbuff, recvbuff, send_count, group_info);
}

template <typename T>
bool CollectiveOpsImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                  const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return Broadcast<T>(sendbuff, recvbuff, count, root, group_info);
}

template <typename T>
bool CollectiveOpsImpl::Send(const void *sendbuff, size_t count, uint32_t root, const ps::core::AbstractNodePtr &node,
                             const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return Send<T>(sendbuff, count, root, group_info);
}

template <typename T>
bool CollectiveOpsImpl::Recv(void *recvbuff, size_t count, uint32_t root, const ps::core::AbstractNodePtr &node,
                             const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return Recv<T>(recvbuff, count, root, group_info);
}

template <typename T>
bool CollectiveOpsImpl::Gather(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                               const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return Gather<T>(sendbuff, recvbuff, count, root, group_info);
}

template <typename T>
bool CollectiveOpsImpl::Scatter(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info) {
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(node, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);

  // Initialize collective communication parameters.
  node_ = node;
  node_role_ = node_->role();
  rank_id_ = node_->rank_id();
  rank_size_ = group_info.size;
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  return Scatter<T>(sendbuff, recvbuff, count, root, group_info);
}

template bool CollectiveOpsImpl::AllReduce<float>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<int64_t>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<int>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<int8_t>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<int16_t>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<float16>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllReduce<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, int reduce_op,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::AllGather<float>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<int64_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<int>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<int8_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<float16>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<bfloat16>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<uint8_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<uint16_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<uint32_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<uint64_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::AllGather<double>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::RingAllGather<float>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                      const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<int64_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                        const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<int>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<int8_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                       const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<float16>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                        const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<bfloat16>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                         const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<uint8_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                        const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<uint16_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                         const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<uint32_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                         const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<uint64_t>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                         const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::RingAllGather<double>(const void *sendbuff, void *recvbuff, size_t send_count,
                                                       const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Broadcast<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const ps::core::AbstractNodePtr &node,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const ps::core::AbstractNodePtr &node,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Broadcast<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                     const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Broadcast<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Send<float>(const void *sendbuff, size_t count, uint32_t root,
                                             const ps::core::AbstractNodePtr &node,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int64_t>(const void *sendbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int>(const void *sendbuff, size_t count, uint32_t root,
                                           const ps::core::AbstractNodePtr &node,
                                           const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int8_t>(const void *sendbuff, size_t count, uint32_t root,
                                              const ps::core::AbstractNodePtr &node,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<float16>(const void *sendbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<bfloat16>(const void *sendbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint8_t>(const void *sendbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint16_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint32_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint64_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<double>(const void *sendbuff, size_t count, uint32_t root,
                                              const ps::core::AbstractNodePtr &node,
                                              const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Send<float>(const void *sendbuff, size_t count, uint32_t root,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int64_t>(const void *sendbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int>(const void *sendbuff, size_t count, uint32_t root,
                                           const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<int8_t>(const void *sendbuff, size_t count, uint32_t root,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<float16>(const void *sendbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<bfloat16>(const void *sendbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint8_t>(const void *sendbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint16_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint32_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<uint64_t>(const void *sendbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Send<double>(const void *sendbuff, size_t count, uint32_t root,
                                              const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Recv<float>(void *recvbuff, size_t count, uint32_t root,
                                             const ps::core::AbstractNodePtr &node,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int64_t>(void *recvbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int>(void *recvbuff, size_t count, uint32_t root,
                                           const ps::core::AbstractNodePtr &node,
                                           const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int8_t>(void *recvbuff, size_t count, uint32_t root,
                                              const ps::core::AbstractNodePtr &node,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<float16>(void *recvbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<bfloat16>(void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint8_t>(void *recvbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint16_t>(void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint32_t>(void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint64_t>(void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<double>(void *recvbuff, size_t count, uint32_t root,
                                              const ps::core::AbstractNodePtr &node,
                                              const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Recv<float>(void *recvbuff, size_t count, uint32_t root,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int64_t>(void *recvbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int>(void *recvbuff, size_t count, uint32_t root,
                                           const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<int8_t>(void *recvbuff, size_t count, uint32_t root,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<float16>(void *recvbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<bfloat16>(void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint8_t>(void *recvbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint16_t>(void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint32_t>(void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<uint64_t>(void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Recv<double>(void *recvbuff, size_t count, uint32_t root,
                                              const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Gather<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                               const ps::core::AbstractNodePtr &node,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const ps::core::AbstractNodePtr &node,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                             const ps::core::AbstractNodePtr &node,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const ps::core::AbstractNodePtr &node,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const ps::core::AbstractNodePtr &node,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Gather<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                               const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                             const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Gather<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Scatter<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const ps::core::AbstractNodePtr &node,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                              const ps::core::AbstractNodePtr &node,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const ps::core::AbstractNodePtr &node,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const ps::core::AbstractNodePtr &node,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const ps::core::AbstractNodePtr &node,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const ps::core::AbstractNodePtr &node,
                                                 const CommunicationGroupInfo &group_info);

template bool CollectiveOpsImpl::Scatter<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                              const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<int8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<float16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<bfloat16>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint8_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint16_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint32_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<uint64_t>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
template bool CollectiveOpsImpl::Scatter<double>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                 const CommunicationGroupInfo &group_info);

}  // namespace server
}  // namespace fl
}  // namespace mindspore
