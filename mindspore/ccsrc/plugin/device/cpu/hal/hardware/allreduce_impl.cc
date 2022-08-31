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

#include "plugin/device/cpu/hal/hardware/allreduce_impl.h"

#include <vector>
#include <functional>
#include <memory>

namespace mindspore {
namespace device {
namespace cpu {
namespace {
constexpr size_t kWaitTimeout = 30;
}  // namespace

bool AllReduceLauncher::Initialize() {
  const auto &cluster_ctx = distributed::cluster::ClusterContext::instance();
  MS_EXCEPTION_IF_NULL(cluster_ctx);
  auto node_base = cluster_ctx->node_base();
  MS_EXCEPTION_IF_NULL(node_base);
  rank_id_ = node_base->rank_id();

  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(node_base);
  abs_node_ = std::make_shared<ps::core::CollectiveNode>(cgn);
  if (!abs_node_->Start()) {
    MS_LOG(ERROR) << "Failed to start the cpu collective node.";
    return false;
  }

  node_role_ = cluster_ctx->node_role();
  rank_size_ = static_cast<size_t>(cluster_ctx->node_num(cluster_ctx->node_role()));
  return true;
}

bool AllReduceLauncher::Finalize() {
  MS_EXCEPTION_IF_NULL(abs_node_);
  if (!abs_node_->Finish()) {
    MS_LOG(WARNING) << "Failed to finish the cpu collective node.";
  }
  if (!abs_node_->Stop()) {
    MS_LOG(ERROR) << "Failed to stop the cpu collective node.";
    return false;
  }
  return true;
}

bool AllReduceLauncher::Execute(const void *input_data, void *const output_data, size_t data_size) const {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(output_data);
  // If node is scheduler, don't need to participate in the reduction.
  if (node_role_ == distributed::kEnvRoleOfScheduler) {
    return true;
  }
  size_t data_num = data_size / sizeof(float);
  if (data_num < rank_size_) {
    MS_LOG(DEBUG) << "AllReduceLauncher executes ReduceBroadcastAllReduce algorithm on the rank " << rank_id_;
    return ReduceBroadcastAllReduce(input_data, output_data, data_size);
  }
  // If the data number is not less than the node number, the RingAllReduce algorithm is used.
  MS_LOG(DEBUG) << "AllReduceLauncher executes RingAllReduce algorithm on the rank " << rank_id_;
  return RingAllReduce(input_data, output_data, data_size);
}

bool AllReduceLauncher::RingAllReduce(const void *input_data, void *const output_data, size_t data_size) const {
  int memcpy_ret = memcpy_s(output_data, data_size, input_data, data_size);
  if (memcpy_ret != EOK) {
    MS_LOG(ERROR) << "RingAllReduce memcpy_s input_data error, errorno(" << memcpy_ret << ")";
    return false;
  }
  MS_EXCEPTION_IF_CHECK_FAIL((rank_size_ != 0), "The rank size is zero.");
  size_t data_num = data_size / sizeof(float);
  size_t chunk_size = data_num / rank_size_;
  size_t remainder_size = data_num % rank_size_;
  std::vector<size_t> chunk_sizes(rank_size_, chunk_size);
  // The rest of the data should be assigned to each chunk.
  for (size_t i = 0; i < remainder_size; i++) {
    chunk_sizes[i]++;
  }
  // Store offsets to get every data chunk's address.
  std::vector<size_t> chunk_offset;
  for (size_t i = 0; i < rank_size_; i++) {
    size_t ofs =
      std::accumulate(chunk_sizes.begin(), chunk_sizes.begin() + SizeToLong(i), size_t(0), std::plus<size_t>());
    chunk_offset.push_back(ofs);
  }

  auto *output_buff = reinterpret_cast<float *>(output_data);
  uint32_t send_to_rank = SizeToUint((rank_id_ + 1) % rank_size_);
  uint32_t rec_from_rank = SizeToUint((rank_id_ - 1 + rank_size_) % rank_size_);
  MS_LOG(DEBUG) << "AllReduce data_num:" << data_num << ", rank_size_:" << rank_size_ << ", rank_id_:" << rank_id_
                << ", chunk_size:" << chunk_size << ", remainder_size:" << remainder_size
                << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", rec_from_rank:" << rec_from_rank;

  // Ring ReduceScatter.
  MS_LOG(DEBUG) << "Start Ring ReduceScatter.";
  MS_EXCEPTION_IF_NULL(abs_node_);
  for (size_t i = 0; i < rank_size_ - 1; i++) {
    // Step 1: Async send data to next rank.
    size_t send_chunk_index = (rank_id_ - i + rank_size_) % rank_size_;
    float *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_req_id = abs_node_->CollectiveSendAsync(ps::core::NodeRole::WORKER, send_to_rank, send_chunk,
                                                      chunk_sizes[send_chunk_index] * sizeof(float));
    // Step 2: Async receive data to next rank and wait until it's done.
    size_t rec_chunk_index = (rank_id_ - i - 1 + rank_size_) % rank_size_;
    float *rec_chunk = output_buff + chunk_offset[rec_chunk_index];
    MS_LOG(DEBUG) << "Ring ReduceScatter send_to_rank:" << send_to_rank << ", rec_from_rank:" << rec_from_rank
                  << ", send data_num:" << chunk_sizes[send_chunk_index]
                  << ", rec data_num:" << chunk_sizes[rec_chunk_index] << ", iteration:" << i;

    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = abs_node_->CollectiveReceiveAsync(ps::core::NodeRole::WORKER, rec_from_rank, &rec_ptr);
    if (!abs_node_->CollectiveWait(rec_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "Ring ReduceScatter wait receiving [" << rec_req_id.first << "," << rec_req_id.second
                    << "] failed.";
      return false;
    }
    // Step 3: Reduce the data, so we can overlap the time cost of send.
    MS_EXCEPTION_IF_NULL(rec_ptr);
    const auto *tmp_data = reinterpret_cast<float *>(rec_ptr->data());
    for (size_t j = 0; j < chunk_sizes[rec_chunk_index]; j++) {
      rec_chunk[j] += tmp_data[j];
    }
    // Step 4: Wait until send is done.
    if (!abs_node_->Wait(send_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "Ring ReduceScatter wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring ReduceScatter.";

  // Ring AllGather.
  MS_LOG(DEBUG) << "Start Ring AllGather.";
  for (size_t i = 0; i < rank_size_ - 1; i++) {
    size_t send_chunk_index = (rank_id_ - i + 1 + rank_size_) % rank_size_;
    float *send_chunk = output_buff + chunk_offset[send_chunk_index];
    auto send_req_id = abs_node_->CollectiveSendAsync(ps::core::NodeRole::WORKER, send_to_rank, send_chunk,
                                                      chunk_sizes[send_chunk_index] * sizeof(float));
    size_t rec_chunk_index = (rank_id_ - i + rank_size_) % rank_size_;
    float *rec_chunk = output_buff + chunk_offset[rec_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << send_to_rank << ", rec_from_rank:" << rec_from_rank
                  << ", send data_num:" << chunk_sizes[send_chunk_index]
                  << ", rec data_num:" << chunk_sizes[rec_chunk_index] << ", iteration:" << i;

    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = abs_node_->CollectiveReceiveAsync(ps::core::NodeRole::WORKER, rec_from_rank, &rec_ptr);
    if (!abs_node_->CollectiveWait(rec_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "Ring AllGather wait receiving " << rec_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(rec_ptr);
    memcpy_ret = memcpy_s(rec_chunk, chunk_sizes[rec_chunk_index] * sizeof(float), rec_ptr->data(), rec_ptr->size());
    if (memcpy_ret != 0) {
      MS_LOG(ERROR) << "Ring AllGather memcpy_s received data error, errorno(" << memcpy_ret << ")";
      return false;
    }
    if (!abs_node_->Wait(send_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "RingAllReduce wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring AllGather.";
  return true;
}

const std::shared_ptr<ps::core::CollectiveNode> &AllReduceLauncher::collective_node() const { return abs_node_; }

bool AllReduceLauncher::ReduceBroadcastAllReduce(const void *input_data, void *const output_data,
                                                 size_t data_size) const {
  int memcpy_ret = memcpy_s(output_data, data_size, input_data, data_size);
  if (memcpy_ret != EOK) {
    MS_LOG(ERROR) << "ReduceBroadcastAllReduce memcpy_s input_data error, errorno(" << memcpy_ret << ")";
    return false;
  }
  size_t data_num = data_size / sizeof(float);
  float *output_buff = reinterpret_cast<float *>(output_data);
  // Reduce data to rank 0 process.
  MS_LOG(DEBUG) << "Start Reduce to rank 0 process.";
  MS_EXCEPTION_IF_NULL(abs_node_);
  if (rank_id_ == 0) {
    for (uint32_t i = 1; i < rank_size_; i++) {
      std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
      MS_LOG(DEBUG) << "Reduce rank 0 receive from rank " << i;
      auto rec_req_id = abs_node_->CollectiveReceiveAsync(ps::core::NodeRole::WORKER, i, &rec_ptr);
      if (!abs_node_->CollectiveWait(rec_req_id, kWaitTimeout)) {
        MS_LOG(ERROR) << "Reduce wait receiving " << rec_req_id << " failed.";
        return false;
      }
      MS_EXCEPTION_IF_NULL(rec_ptr);
      const auto *tmp_data = reinterpret_cast<float *>(rec_ptr->data());
      for (size_t j = 0; j < data_num; j++) {
        output_buff[j] += tmp_data[j];
      }
    }
  } else {
    MS_LOG(DEBUG) << "Reduce send data to rank 0 process.";
    auto send_req_id =
      abs_node_->CollectiveSendAsync(ps::core::NodeRole::WORKER, 0, input_data, data_num * sizeof(float));
    if (!abs_node_->Wait(send_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "Reduce wait sending " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Reduce.";

  // Broadcast data to not rank 0 process.
  MS_LOG(DEBUG) << "Start broadcast from rank 0 to other processes.";
  if (rank_id_ == 0) {
    for (uint32_t i = 1; i < rank_size_; i++) {
      MS_LOG(DEBUG) << "Broadcast data to process " << i;
      auto send_req_id =
        abs_node_->CollectiveSendAsync(ps::core::NodeRole::WORKER, i, output_buff, data_num * sizeof(float));
      if (!abs_node_->Wait(send_req_id, kWaitTimeout)) {
        MS_LOG(ERROR) << "Broadcast wait sending " << send_req_id << " failed.";
        return false;
      }
    }
  } else {
    MS_LOG(DEBUG) << "Broadcast receive from rank 0.";
    std::shared_ptr<std::vector<unsigned char>> rec_ptr = nullptr;
    auto rec_req_id = abs_node_->CollectiveReceiveAsync(ps::core::NodeRole::WORKER, 0, &rec_ptr);
    if (!abs_node_->CollectiveWait(rec_req_id, kWaitTimeout)) {
      MS_LOG(ERROR) << "Broadcast wait receiving " << rec_req_id << " failed.";
      return false;
    }
    MS_EXCEPTION_IF_NULL(rec_ptr);
    memcpy_ret = memcpy_s(output_buff, data_num * sizeof(float), rec_ptr->data(), rec_ptr->size());
    if (memcpy_ret != 0) {
      MS_LOG(ERROR) << "Broadcast memcpy_s received data error, errorno(" << memcpy_ret << ")";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End broadcast.";
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
