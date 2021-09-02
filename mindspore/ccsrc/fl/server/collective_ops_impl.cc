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

#include "fl/server/collective_ops_impl.h"

namespace mindspore {
namespace fl {
namespace server {
void CollectiveOpsImpl::Initialize(const std::shared_ptr<ps::core::ServerNode> &server_node) {
  MS_EXCEPTION_IF_NULL(server_node);
  server_node_ = server_node;
  local_rank_ = server_node_->rank_id();
  server_num_ = ps::PSContext::instance()->initial_server_num();
  return;
}

template <typename T>
bool CollectiveOpsImpl::RingAllReduce(const void *sendbuff, void *recvbuff, size_t count) {
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  int ret = memcpy_s(recvbuff, count * sizeof(T), sendbuff, count * sizeof(T));
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
    return false;
  }

  uint32_t rank_size = server_num_;
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
  uint32_t send_to_rank = (local_rank_ + 1) % rank_size;
  uint32_t recv_from_rank = (local_rank_ - 1 + rank_size) % rank_size;
  MS_LOG(DEBUG) << "AllReduce count:" << count << ", rank_size:" << rank_size << ", local_rank_:" << local_rank_
                << ", chunk_size:" << chunk_size << ", remainder_size:" << remainder_size
                << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", recv_from_rank:" << recv_from_rank;

  // Ring ReduceScatter.
  MS_LOG(DEBUG) << "Start Ring ReduceScatter.";
  std::unique_ptr<T[]> tmp_recv_chunk = std::make_unique<T[]>(chunk_sizes[0]);
  MS_EXCEPTION_IF_NULL(tmp_recv_chunk);
  for (size_t i = 0; i < rank_size - 1; i++) {
    // Step 1: Async send data to next rank.
    size_t send_chunk_index = (local_rank_ - i + rank_size) % rank_size;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_req_id = server_node_->CollectiveSendAsync(ps::core::NodeRole::SERVER, send_to_rank, send_chunk,
                                                         chunk_sizes[send_chunk_index] * sizeof(T));
    // Step 2: Async receive data to next rank and wait until it's done.
    size_t recv_chunk_index = (local_rank_ - i - 1 + rank_size) % rank_size;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring ReduceScatter send_to_rank:" << send_to_rank << ", recv_from_rank:" << recv_from_rank
                  << ", send count:" << chunk_sizes[send_chunk_index]
                  << ", recv count:" << chunk_sizes[recv_chunk_index] << ", iteration:" << i;

    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = server_node_->CollectiveReceiveAsync(ps::core::NodeRole::SERVER, recv_from_rank, &recv_str);
    if (!server_node_->CollectiveWait(recv_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    ret = memcpy_s(tmp_recv_chunk.get(), chunk_sizes[recv_chunk_index] * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }

    // Step 3: Reduce the data so we can overlap the time cost of send.
    for (size_t j = 0; j < chunk_sizes[recv_chunk_index]; j++) {
      recv_chunk[j] += tmp_recv_chunk[j];
    }
    // Step 4: Wait until send is done.
    if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring ReduceScatter.";

  // Ring AllGather.
  MS_LOG(DEBUG) << "Start Ring AllGather.";
  for (size_t i = 0; i < rank_size - 1; i++) {
    size_t send_chunk_index = (local_rank_ - i + 1 + rank_size) % rank_size;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_req_id = server_node_->CollectiveSendAsync(ps::core::NodeRole::SERVER, send_to_rank, send_chunk,
                                                         chunk_sizes[send_chunk_index] * sizeof(T));
    size_t recv_chunk_index = (local_rank_ - i + rank_size) % rank_size;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << send_to_rank << ", recv_from_rank:" << recv_from_rank
                  << ", send count:" << chunk_sizes[send_chunk_index]
                  << ", recv count:" << chunk_sizes[recv_chunk_index] << ", iteration:" << i;

    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = server_node_->CollectiveReceiveAsync(ps::core::NodeRole::SERVER, recv_from_rank, &recv_str);
    if (!server_node_->CollectiveWait(recv_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    ret = memcpy_s(recv_chunk, chunk_sizes[recv_chunk_index] * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
    if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring AllGather.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::ReduceBroadcastAllReduce(const void *sendbuff, void *recvbuff, size_t count) {
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  uint32_t rank_size = server_num_;
  MS_LOG(DEBUG) << "Reduce Broadcast AllReduce rank_size:" << rank_size << ", local_rank_:" << local_rank_
                << ", count:" << count;
  int ret = memcpy_s(recvbuff, count * sizeof(T), sendbuff, count * sizeof(T));
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
    return false;
  }
  T *output_buff = reinterpret_cast<T *>(recvbuff);
  // Reduce data to rank 0 process.
  MS_LOG(DEBUG) << "Start Reduce to rank 0 process.";
  if (local_rank_ == 0) {
    std::unique_ptr<T[]> tmp_recv_buff = std::make_unique<T[]>(count);
    MS_EXCEPTION_IF_NULL(tmp_recv_buff);
    for (uint32_t i = 1; i < rank_size; i++) {
      std::shared_ptr<std::vector<unsigned char>> recv_str;
      MS_LOG(DEBUG) << "Reduce rank 0 receive from rank " << i;
      auto recv_req_id = server_node_->CollectiveReceiveAsync(ps::core::NodeRole::SERVER, i, &recv_str);
      if (!server_node_->CollectiveWait(recv_req_id, kCollectiveCommTimeout)) {
        MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
        return false;
      }
      ret = memcpy_s(tmp_recv_buff.get(), count * sizeof(T), recv_str->data(), recv_str->size());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
      for (size_t j = 0; j < count; j++) {
        output_buff[j] += tmp_recv_buff[j];
      }
    }
  } else {
    MS_LOG(DEBUG) << "Reduce send data to rank 0 process.";
    auto send_req_id = server_node_->CollectiveSendAsync(ps::core::NodeRole::SERVER, 0, sendbuff, count * sizeof(T));
    if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Reduce.";

  // Broadcast data to not 0 rank process.
  MS_LOG(DEBUG) << "Start broadcast from rank 0 to other processes.";
  if (local_rank_ == 0) {
    for (uint32_t i = 1; i < rank_size; i++) {
      MS_LOG(DEBUG) << "Broadcast data to process " << i;
      auto send_req_id =
        server_node_->CollectiveSendAsync(ps::core::NodeRole::SERVER, i, output_buff, count * sizeof(T));
      if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
        MS_LOG(ERROR) << "CollectiveWait " << send_req_id << " failed.";
        return false;
      }
    }
  } else {
    MS_LOG(DEBUG) << "Broadcast receive from rank 0.";
    std::shared_ptr<std::vector<unsigned char>> recv_str;
    auto recv_req_id = server_node_->CollectiveReceiveAsync(ps::core::NodeRole::SERVER, 0, &recv_str);
    if (!server_node_->CollectiveWait(recv_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveWait " << recv_req_id << " failed.";
      return false;
    }
    ret = memcpy_s(output_buff, count * sizeof(T), recv_str->data(), recv_str->size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End broadcast.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count) {
  // The collective communication API does not support calling Send and Recv concurrently with multiple threads;
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  uint32_t rank_size = server_num_;
  if (count >= rank_size) {
    return RingAllReduce<T>(sendbuff, recvbuff, count);
  } else {
    return ReduceBroadcastAllReduce<T>(sendbuff, recvbuff, count);
  }
}

bool CollectiveOpsImpl::ReInitForScaling() {
  // If CollectiveOpsImpl is not initialized yet but the scaling event is triggered, do not throw exception.
  if (server_node_ == nullptr) {
    return true;
  }

  MS_LOG(INFO) << "Cluster scaling out completed. Reinitialize ring for collective communication.";
  local_rank_ = server_node_->rank_id();
  server_num_ = IntToUint(server_node_->server_num());
  MS_LOG(INFO) << "After scheduler scaling out, this server's rank is " << local_rank_ << ", server number is "
               << server_num_;
  return true;
}

template bool CollectiveOpsImpl::RingAllReduce<float>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::RingAllReduce<size_t>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::RingAllReduce<int>(const void *sendbuff, void *recvbuff, size_t count);

template bool CollectiveOpsImpl::ReduceBroadcastAllReduce<float>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::ReduceBroadcastAllReduce<size_t>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::ReduceBroadcastAllReduce<int>(const void *sendbuff, void *recvbuff, size_t count);

template bool CollectiveOpsImpl::AllReduce<float>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::AllReduce<size_t>(const void *sendbuff, void *recvbuff, size_t count);
template bool CollectiveOpsImpl::AllReduce<int>(const void *sendbuff, void *recvbuff, size_t count);
}  // namespace server
}  // namespace fl
}  // namespace mindspore
