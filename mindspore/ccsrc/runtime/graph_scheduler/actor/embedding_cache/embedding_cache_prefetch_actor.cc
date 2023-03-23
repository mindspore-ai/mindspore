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

#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"
#include <limits>
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "kernel/common_utils.h"
#include "runtime/graph_scheduler/actor/rpc/rpc_actor.h"
#include "proto/topology.pb.h"
#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/rpc/tcp/constants.h"
#include "runtime/graph_scheduler/actor/embedding_cache/device_dense_embedding_operation.h"
#include "runtime/graph_scheduler/actor/embedding_cache/device_sparse_embedding_operation.h"

namespace mindspore {
namespace runtime {
using distributed::cluster::ClusterContext;
using mindspore::session::KernelGraph;

namespace {
// Generate unique inter process edge name, format:
// src role + src rank id -> dst role + dst rank id + embedding cache operation + parameter key.
std::string GenerateInterProcessEdge(const std::string &src_role, uint32_t src_rank, const std::string &dst_role,
                                     uint32_t dst_rank, const std::string &cache_operation, int32_t param_key) {
  std::string edge = src_role + std::to_string(src_rank) + "->" + dst_role + std::to_string(dst_rank) + "_" +
                     cache_operation + "_" + distributed::kParameterKey + std::to_string(param_key);
  return edge;
}

ActorRouteTableProxyPtr CreateRouteTableProxy() {
  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    ClusterContext::instance()->node_base());
  ActorRouteTableProxyPtr actor_route_table_proxy = std::make_shared<ActorRouteTableProxy>(cgn);
  MS_EXCEPTION_IF_NULL(actor_route_table_proxy);
  return actor_route_table_proxy;
}

// Create a sender and receiver pair,The sender and receiver are paired.
// When creating a sender, need to create and specify the receiver paired with it in advance.
SendRecvPair CreateSenderReceiverPair(uint32_t worker_rank, uint32_t server_rank, const std::string &cache_operation,
                                      int32_t param_key, device::DeviceContext *cpu_device_context) {
  // Create sender and receiver pair.
  ReceiverPtr receiver = std::make_shared<Receiver>(cpu_device_context);
  MS_EXCEPTION_IF_NULL(receiver);
  SenderPtr sender = std::make_shared<Sender>(cpu_device_context);
  MS_EXCEPTION_IF_NULL(sender);
  sender->set_receiver(receiver);

  // Set inter process edge
  receiver->set_inter_process_edge_name(GenerateInterProcessEdge(distributed::kEnvRoleOfPServer, server_rank,
                                                                 distributed::kEnvRoleOfWorker, worker_rank,
                                                                 cache_operation, param_key));
  sender->set_inter_process_edge_name(GenerateInterProcessEdge(distributed::kEnvRoleOfWorker, worker_rank,
                                                               distributed::kEnvRoleOfPServer, server_rank,
                                                               cache_operation, param_key));

  // Set route table proxy.
  receiver->set_actor_route_table_proxy(CreateRouteTableProxy());
  sender->set_actor_route_table_proxy(CreateRouteTableProxy());

  return std::make_pair(sender, receiver);
}

// Get cache operation service id which is used to decide which set of cache services to request.
// The server side executes the corresponding service according to this id.
int32_t GetCacheOpsServiceId(const std::string &cache_operation, int32_t param_key) {
  static mindspore::HashMap<std::string, int32_t> cache_ops_to_index;
  if (cache_ops_to_index.empty()) {
    int32_t cnt = 0;
    for (const auto &cache_op : distributed::kEmbeddingCacheOps) {
      cache_ops_to_index[cache_op] = cnt++;
    }
  }

  auto iter = cache_ops_to_index.find(cache_operation);
  if (iter == cache_ops_to_index.end()) {
    MS_LOG(EXCEPTION) << "Can not find index for cache operation: " << cache_operation;
  }

  int32_t id = SizeToInt(distributed::kEmbeddingCacheOps.size()) * param_key + iter->second;
  return id;
}
}  // namespace

void EmbeddingCachePrefetchActor::Initialize() {
  if (initialized_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
  if (!device_context_->device_res_manager_->CreateStream(&stream_id_)) {
    MS_LOG(EXCEPTION) << "Create stream failed.";
  }

  // Create and Initialize the random number generator for embedding values.
  const std::uint64_t seed = 0;
  const size_t skip = 0;
  rnd_gen_ = std::make_unique<distributed::RandomGenerator<DataType, Generator, Distribution>>(seed, skip);

  const double mean = 0.0;
  const double sigma = 0.01;
  (void)rnd_gen_->Initialize(mean, sigma);

  // Get embedding cache table info.
  local_host_cache_size_ = embedding_cache_table_manager.host_cache_size_;
  vocab_size_ = embedding_cache_table_manager.vocab_size_;
  local_embedding_slice_bounds_ = embedding_cache_table_manager.local_embedding_slice_bounds_;
  local_device_cache_bounds_ = embedding_cache_table_manager.local_device_cache_bounds_;

  // Initialize CPU device context. The origin device context for embedding cache prefetch actor is GPU or NPU. But we
  // still need the CPU device context to allocate host memory.
  device::DeviceContextKey host_key = {"CPU", 0};
  cpu_device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  cpu_device_context_->Initialize();

  server_num_ = PSContext::instance()->server_num();
  if (server_num_ == 0) {
    MS_LOG(EXCEPTION) << "The number of servers is at least 1, but get 0";
  }

  // Build and link rpc operators.
  BuildRpcOperators();
  LinkRpcOperators();

  // Create the device embedding operation.
  if (!distributed::EmbeddingCacheTableManager::GetInstance().is_sparse_format()) {
    emb_ops_ = new DeviceDenseEmbeddingOperation(this, device_context_, local_embedding_slice_bounds_,
                                                 local_device_cache_bounds_, &statistics_info_, stream_id_);
  } else {
    emb_ops_ = new DeviceSparseEmbeddingOperation(this, device_context_, local_embedding_slice_bounds_,
                                                  local_device_cache_bounds_, &statistics_info_, stream_id_);
  }
  MS_EXCEPTION_IF_NULL(emb_ops_);
  if (!emb_ops_->Initialize()) {
    MS_LOG(ERROR) << "Failed to initialize the device embedding operation.";
  }

  // Get the id range of each server's embedding table slice.
  emb_ops_->GetRemoteEmbeddingSliceBound(vocab_size_, server_num_, &remote_embedding_slice_bounds_);

  initialized_ = true;
}

void EmbeddingCachePrefetchActor::Finalize(bool finalize_remote) {
  std::lock_guard<std::mutex> lock(finalize_mutex_);
  if (!initialized_ || finalized_) {
    return;
  }

  running_ = false;
  PsDataPrefetch::GetInstance().NotifyFinalize();

  if (finalize_remote) {
    (void)FinalizeRemote();
  }

  data_parser_.notify_all();

  if (emb_ops_ != nullptr) {
    delete emb_ops_;
    emb_ops_ = nullptr;
  }

  if (rnd_gen_ != nullptr) {
    (void)rnd_gen_->Finalize();
  }

  rpc_operators_.clear();
  finalized_ = true;
  initialized_ = false;
}

void EmbeddingCachePrefetchActor::IncreaseGraphStep(const std::string &channel_name) {
  if (!running_) {
    std::string error_info =
      !error_info_.empty() ? error_info_ : "Embedding cache prefetch actor is finalized abnormally.";
    MS_LOG(EXCEPTION) << error_info;
  }
  if (graph_step_ >= UINT64_MAX) {
    MS_LOG(EXCEPTION) << "The graph step(" << graph_step_ << ") will exceed the maximum value of uint64_t.";
  }
  if (graph_step_ == 0) {
    MS_LOG(INFO) << "Graph running waiting embedding table init begin:" << finish_init_parameters_on_remote_;
    std::unique_lock<std::mutex> locker(data_mutex_);
    data_parser_.wait(locker, [this] { return ((finish_init_parameters_on_remote_ == true) || (running_ == false)); });
    if (!running_) {
      std::string error_info =
        !error_info_.empty() ? error_info_ : "Embedding cache prefetch actor is finalized abnormally.";
      MS_LOG(EXCEPTION) << error_info;
    }
    MS_LOG(INFO) << "Graph running waiting embedding table init end.";
  }
  graph_step_++;
  set_channel_name(channel_name);
  if (!PsDataPrefetch::GetInstance().TryWakeChannel(channel_name)) {
    MS_LOG(EXCEPTION) << "TryWakeChannel failed, channel name: " << channel_name;
  }
  data_parser_.notify_one();
}

void EmbeddingCachePrefetchActor::Run() {
  running_ = true;

  // Bind device to current thread to gain device control privileges
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
  if (!device_context_->device_res_manager_->BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Failed to bind device to current thread.";
    running_ = false;
    PsDataPrefetch::GetInstance().NotifyFinalize();
    return;
  }

  // Wait initialize parameters on remote.
  // Prevents the subsequent prefetch cache from failing due to the long initialization time of the large parameter on
  // the remote side.
  WaitInitParametersOnRemote();

  // Wait data channel ready.
  WaitDataChannelInit();

  MS_LOG(INFO) << "Begin prefetching cache.";
  while (running_) {
    if (!PrefetchCache()) {
      running_ = false;
      // If prefetch cache failed, need to finalize data prefetch thread which is executing
      // PsDataPrefetch::PrefetchData(), so as to the minddata can release resource normally.
      PsDataPrefetch::GetInstance().NotifyFinalize();
    }
  }
  MS_LOG(INFO) << "End prefetching cache.";
}

bool EmbeddingCachePrefetchActor::PrefetchCache() {
  // 1. Acquire batch ids
  void *data = nullptr;
  RETURN_IF_FALSE_WITH_LOG(PsDataPrefetch::GetInstance().QueryData(channel_name_, &data), "Query input data failed.");

  if (data == nullptr) {
    MS_LOG(INFO) << "There is no input data of dataset channel name:" << channel_name_;
    std::unique_lock<std::mutex> locker(data_mutex_);
    const int64_t longest_time_to_wait = 100;
    (void)data_parser_.wait_for(locker, std::chrono::milliseconds(longest_time_to_wait));
    return true;
  }

  RETURN_IF_FALSE_WITH_LOG(IncreaseStep(), "Increase step failed.");
  auto data_size = PsDataPrefetch::GetInstance().data_size(channel_name_);
  if (data_size == 0) {
    MS_LOG(ERROR) << "The data size of batch ids can not be zero.";
    return false;
  }
  auto batch_ids = reinterpret_cast<int *>(data);
  auto batch_ids_num = data_size / sizeof(int);
  auto ret = memset_s(&statistics_info_, sizeof(statistics_info_), 0, sizeof(statistics_info_));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset for cache statistics info failed, errno[" << ret << "]";
    return false;
  }

  // 2. Count cache miss ids.
  RETURN_IF_FALSE_WITH_LOG(emb_ops_->CountCacheMissIds(batch_ids, batch_ids_num, data_step_, graph_running_step_,
                                                       &device_cache_need_wait_graph_, &host_cache_need_wait_graph_),
                           "Count cache miss ids failed.");

  if ((device_cache_need_wait_graph_ || host_cache_need_wait_graph_) && (!WaitGraphRun())) {
    MS_LOG(ERROR) << "Cache prefetching waits graph finish failed.";
    return false;
  }

  // 3. If the device cache does not reach 100% hit rate, the cache needs to be updated.
  RETURN_IF_FALSE_WITH_LOG(UpdateCache(), "Update local cache failed.");

  RETURN_IF_FALSE_WITH_LOG(PsDataPrefetch::GetInstance().FinalizeData(channel_name_), "Finalize data failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::IncreaseStep() {
  if (data_step_ >= UINT64_MAX) {
    MS_LOG(ERROR) << "The data step (" << data_step_ << ") will exceed the maximum value of uint64_t.";
    return false;
  }
  data_step_++;
  set_current_graph_step();
  if (graph_running_step_ > data_step_) {
    MS_LOG(ERROR) << "The graph running step (" << graph_running_step_ << ") exceed the data step (" << data_step_
                  << ").";
    return false;
  }
  return true;
}

bool EmbeddingCachePrefetchActor::WaitGraphRun() {
  MS_LOG(INFO) << "Hash table has no space to insert new data and retries within 2 minutes.";
  std::unique_lock<std::mutex> locker(data_mutex_);
  const int64_t longest_time_to_wait = 120;
  if (!data_parser_.wait_for(locker, std::chrono::seconds(longest_time_to_wait),
                             [this] { return graph_step_ > graph_running_step_; })) {
    std::string err_info = "Prefetch embedding cache timeout, please enlarge the vocab cache size(graph step:" +
                           std::to_string(graph_step_) + ", graph running step:" + std::to_string(graph_running_step_) +
                           ").";
    SetErrorInfo(err_info);
    MS_LOG(ERROR) << err_info;
    return false;
  }
  set_current_graph_step();
  return true;
}

bool EmbeddingCachePrefetchActor::ResetEmbeddingHashMap() {
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  const auto &device_hash_map = embedding_cache_table_manager.embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);
  const auto &host_hash_map = embedding_cache_table_manager.embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  device_hash_map->Reset();
  host_hash_map->Reset();
  device_cache_need_wait_graph_ = false;
  host_cache_need_wait_graph_ = false;
  return true;
}

bool EmbeddingCachePrefetchActor::UpdateCache() {
  for (const auto &item : embedding_cache_table_manager.hash_tables_) {
    auto hash_info = item.second;
    RETURN_IF_FALSE_WITH_LOG(PushCacheFromLocalHostToRemote(hash_info), "Push cache from local host to remote failed.");
    RETURN_IF_FALSE_WITH_LOG(emb_ops_->PushCacheFromDeviceToLocalHost(hash_info),
                             "Push cache from device to local host failed.");
    RETURN_IF_FALSE_WITH_LOG(InitLocalCacheForNewIds(hash_info),
                             "Initialize the local cache values using random generator.");
    RETURN_IF_FALSE_WITH_LOG(PullCacheFromRemoteToLocalHost(hash_info), "Pull cache from remote to local host failed.");
    RETURN_IF_FALSE_WITH_LOG(emb_ops_->PullCacheFromLocalHostToDevice(hash_info),
                             "Pull cache from local host to device failed.");
  }
  return true;
}

void EmbeddingCachePrefetchActor::LookupEmbeddingTable(size_t indices_num, size_t embedding_size, size_t first_dim_size,
                                                       const float *input_addr, const int *indices_addr,
                                                       float *output_addr) {
  MS_ERROR_IF_NULL_WO_RET_VAL(input_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(indices_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(output_addr);

  auto type_size = sizeof(float);
  size_t lens = embedding_size * type_size;
  for (size_t i = 0; i < indices_num; ++i) {
    int index = indices_addr[i];
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = IntToSize(index) * embedding_size;
      auto ret = memcpy_s(output_addr, (indices_num - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
        running_ = false;
        return;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_num - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
        running_ = false;
        return;
      }
    }
    output_addr += embedding_size;
  }
}

bool EmbeddingCachePrefetchActor::PushCacheFromLocalHostToRemote(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.host_to_server_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);
  auto host_to_server_ids = embedding_cache_table_manager.embedding_host_cache_->host_to_server_ids.get();
  MS_ERROR_IF_NULL(host_to_server_ids);
  auto host_to_server_index = embedding_cache_table_manager.embedding_host_cache_->host_to_server_index.get();
  MS_ERROR_IF_NULL(host_to_server_index);

  std::vector<float> swap_out_data;
  auto embedding_size = hash_info.embedding_size;
  swap_out_data.resize(swap_indices_size * embedding_size);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());

  RETURN_IF_FALSE_WITH_LOG(LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                host_to_server_index, swap_out_data.data()),
                           "Lookup local host cache failed.");
  RETURN_IF_FALSE_WITH_LOG(PushEmbeddingsToRemote(hash_info.param_key_, host_to_server_ids, swap_indices_size,
                                                  swap_out_data.data(), swap_out_data.size() * sizeof(float)),
                           "Push embeddings to remote failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::PullCacheFromRemoteToLocalHost(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_.server_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);
  auto server_to_host_ids = embedding_cache_table_manager.embedding_host_cache_->server_to_host_ids.get();
  MS_ERROR_IF_NULL(server_to_host_ids);
  auto server_to_host_index = embedding_cache_table_manager.embedding_host_cache_->server_to_host_index.get();
  MS_ERROR_IF_NULL(server_to_host_index);

  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  MS_ERROR_IF_NULL(host_hash_table_addr);
  auto embedding_size = hash_info.embedding_size;
  std::vector<float> lookup_result(swap_indices_size * embedding_size, 0);

  RETURN_IF_FALSE_WITH_LOG(
    PullEembeddingsFromRemote(hash_info.param_key_, server_to_host_ids, swap_indices_size, &lookup_result),
    "Pull embedding from remote failed.");
  RETURN_IF_FALSE_WITH_LOG(InsertLocalHostCache(embedding_size, IntToSize(swap_indices_size), server_to_host_index,
                                                lookup_result.data(), host_hash_table_addr),
                           "Insert local host cache failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::InitLocalCacheForNewIds(const HashTableInfo &hash_info) {
  auto new_id_size = statistics_info_.new_id_size_;
  if (new_id_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);
  auto new_id_index = embedding_cache_table_manager.embedding_host_cache_->new_id_index.get();
  MS_ERROR_IF_NULL(new_id_index);

  // Compute the feature values size needed to be initialized.
  auto embedding_size = hash_info.embedding_size;
  auto total_size = new_id_size * embedding_size;
  std::vector<float> init_result(total_size, 0);

  // Initialize accumulate values with the configured constant value.
  if (hash_info.param_init_info_.param_type_ == distributed::ParamType::kAccumulation) {
    auto init_value = hash_info.param_init_info_.init_val_;
    for (size_t i = 0; i < total_size; ++i) {
      init_result[i] = init_value;
    }
  } else {
    // Initialize embedding values from local random generator for feature ids that have never been seen before.
    for (size_t i = 0; i < total_size; ++i) {
      init_result[i] = rnd_gen_->Next();
    }
  }

  // Insert initialized feature values into the local hash cache.
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  MS_ERROR_IF_NULL(host_hash_table_addr);
  RETURN_IF_FALSE_WITH_LOG(InsertLocalHostCache(embedding_size, IntToSize(new_id_size), new_id_index,
                                                init_result.data(), host_hash_table_addr),
                           "Insert local host cache failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::InsertLocalHostCache(size_t embedding_size, size_t insert_indices_size,
                                                       const int *insert_indices, const float *insert_data,
                                                       float *hash_table_addr) {
  MS_ERROR_IF_NULL(insert_indices);
  MS_ERROR_IF_NULL(insert_data);
  MS_ERROR_IF_NULL(hash_table_addr);

  size_t first_dim_size = local_host_cache_size_;
  size_t thread_num = insert_indices_size / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t proc_len = (insert_indices_size + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t offset = 0;

  auto insert_cache_func = [this](size_t insert_indices_size, size_t embedding_size, size_t first_dim_size,
                                  const int *insert_indices, const float *insert_data, float *hash_table_addr) {
    auto type_size = sizeof(float);
    size_t copy_len = embedding_size * type_size;
    size_t dest_len = copy_len;
    for (size_t i = 0; i < insert_indices_size; ++i) {
      int index = insert_indices[i];
      if (index >= 0 && index < SizeToInt(first_dim_size)) {
        auto ret =
          memcpy_s(hash_table_addr + index * embedding_size, dest_len, insert_data + i * embedding_size, copy_len);
        if (ret != EOK) {
          MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
          running_ = false;
          return;
        }
      }
    }
  };

  for (; i < thread_num; i++) {
    if (offset >= insert_indices_size) {
      break;
    }
    threads[i] = std::thread(insert_cache_func, proc_len, embedding_size, first_dim_size, insert_indices + offset,
                             insert_data + offset * embedding_size, hash_table_addr);
    offset += proc_len;
    if (offset + proc_len > insert_indices_size) {
      proc_len = insert_indices_size - offset;
    }
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

bool EmbeddingCachePrefetchActor::LookupLocalHostCache(size_t embedding_size, size_t indices_num,
                                                       const float *hash_table_addr, const int *indices_addr,
                                                       float *output_addr) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(indices_addr);
  MS_ERROR_IF_NULL(output_addr);

  size_t first_dim_size = local_host_cache_size_;
  size_t thread_num = indices_num / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t proc_len = (indices_num + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t offset = 0;

  for (; i < thread_num; i++) {
    if (offset >= indices_num) {
      break;
    }
    threads[i] =
      std::thread(&EmbeddingCachePrefetchActor::LookupEmbeddingTable, this, proc_len, embedding_size, first_dim_size,
                  hash_table_addr, indices_addr + offset, output_addr + offset * embedding_size);
    offset += proc_len;
    if (offset + proc_len > indices_num) {
      proc_len = indices_num - offset;
    }
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

bool EmbeddingCachePrefetchActor::PullEembeddingsFromRemote(int32_t param_key, const int *ids, size_t ids_num,
                                                            std::vector<float> *outputs) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(outputs);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  std::vector<std::vector<int>> slice_ids_list(server_num_);
  // 1. Partition ids by remote embedding slice bound and get unique ids.
  RETURN_IF_FALSE_WITH_LOG(PartitionIds(ids, ids_num, &slice_ids_list), "Partition ids failed.");

  size_t embedding_dim = outputs->size() / ids_num;
  for (size_t i = 0; i < server_num_; i++) {
    auto &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }

    // 2. Send unique ids to remote to do embedding lookup.
    RETURN_IF_FALSE_WITH_LOG(SendToRemote(distributed::kLookupEmbeddingCache, param_key, i, embedding_dim,
                                          slice_ids.data(), slice_ids.size() * sizeof(int), nullptr, 0, false, false),
                             "Send ids to server failed.");
  }

  std::vector<std::unique_ptr<std::vector<char>>> slice_embeddings_list(server_num_);
  for (size_t i = 0; i < server_num_; i++) {
    if (slice_ids_list[i].empty()) {
      continue;
    }

    // 3. Wait embeddings result.
    slice_embeddings_list[i] = ReceiveFromRemote(distributed::kLookupEmbeddingCache, param_key, i);
    MS_ERROR_IF_NULL(slice_embeddings_list[i]);
    // Received embedding integrity check.
    size_t expected_embedding_size = SizetMulWithOverflowCheck(slice_ids_list[i].size(), embedding_dim);
    size_t received_embedding_size = slice_embeddings_list[i]->size() / sizeof(float);
    if (received_embedding_size != expected_embedding_size) {
      MS_LOG(ERROR) << "Received embedding data from remote is incomplete, expected embedding size: "
                    << expected_embedding_size << ", but received embedding size: " << received_embedding_size;
      return false;
    }
  }

  // 4. Retrieve embeddings by input ids order.
  RETURN_IF_FALSE_WITH_LOG(RetrieveEmbeddings(ids, ids_num, slice_ids_list, slice_embeddings_list, outputs),
                           "Retrieve embeddings failed.");

  return true;
}

bool EmbeddingCachePrefetchActor::PushEmbeddingsToRemote(int32_t param_key, const int *ids, size_t ids_num,
                                                         const float *embeddings, size_t embeddings_len) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embeddings);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  std::vector<std::vector<int>> slice_ids_list(server_num_);
  std::vector<std::vector<float>> slice_embeddings_list(server_num_);
  // 1. Partition ids end embeddings by remote embedding slice bound.
  RETURN_IF_FALSE_WITH_LOG(
    PartitionIdsAndEmbeddings(ids, ids_num, embeddings, embeddings_len, &slice_ids_list, &slice_embeddings_list),
    "Partition ids and embeddings failed.");

  size_t embedding_dim = (embeddings_len / ids_num) / sizeof(float);
  for (size_t i = 0; i < server_num_; i++) {
    auto &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }

    // 2. Send embeddings to remote.
    auto &slice_embeddings = slice_embeddings_list[i];
    RETURN_IF_FALSE_WITH_LOG(
      SendToRemote(distributed::kUpdateEmbeddingCache, param_key, i, embedding_dim, slice_ids.data(),
                   slice_ids.size() * sizeof(int), slice_embeddings.data(), slice_embeddings.size() * sizeof(float)),
      "Send ids and embeddings to server failed.");
  }

  return true;
}

bool EmbeddingCachePrefetchActor::PartitionIds(const int *ids, size_t ids_num,
                                               std::vector<std::vector<int>> *slice_ids_list) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(slice_ids_list);

  for (size_t i = 0; i < slice_ids_list->size(); i++) {
    int begin = SizeToInt(remote_embedding_slice_bounds_[i].first);
    int end = SizeToInt(remote_embedding_slice_bounds_[i].second);

    mindspore::HashSet<int> unique_ids;
    (void)std::for_each(ids, ids + ids_num, [&](int id) {
      if (id >= begin && id <= end) {
        (void)unique_ids.insert(id);
      }
    });

    std::vector<int> &slice_ids = slice_ids_list->at(i);
    (void)std::for_each(unique_ids.begin(), unique_ids.end(), [&](int id) { slice_ids.push_back(id); });
  }

  return true;
}

bool EmbeddingCachePrefetchActor::PartitionIdsAndEmbeddings(const int *ids, size_t ids_num, const float *embeddings,
                                                            size_t embeddings_len,
                                                            std::vector<std::vector<int>> *slice_ids_list,
                                                            std::vector<std::vector<float>> *slice_embeddings_list) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embeddings);
  MS_ERROR_IF_NULL(slice_ids_list);
  MS_ERROR_IF_NULL(slice_embeddings_list);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  size_t embedding_dim = (embeddings_len / ids_num) / sizeof(float);
  size_t partition_num = slice_ids_list->size();
  for (size_t i = 0; i < partition_num; i++) {
    int begin = SizeToInt(remote_embedding_slice_bounds_[i].first);
    int end = SizeToInt(remote_embedding_slice_bounds_[i].second);

    std::vector<int> &slice_ids = slice_ids_list->at(i);
    std::vector<float> &slice_embeddings = slice_embeddings_list->at(i);
    // Ids range offset for multi server.
    int offset = SizeToInt(remote_embedding_slice_bounds_.at(i).first);
    for (size_t j = 0; j < ids_num; j++) {
      if (ids[j] >= begin && ids[j] <= end) {
        slice_ids.push_back(ids[j] - offset);
        (void)slice_embeddings.insert(slice_embeddings.end(), embeddings + (j * embedding_dim),
                                      embeddings + (j * embedding_dim) + embedding_dim);
      }
    }
  }
  return true;
}

bool EmbeddingCachePrefetchActor::SendToRemote(const std::string &cache_operation, int32_t param_key,
                                               size_t server_rank_id, size_t embedding_dim, const void *keys,
                                               size_t keys_len, const void *values, size_t values_len,
                                               bool finalize_remote, bool sync) {
  MS_ERROR_IF_NULL(keys);
  // Find sender corresponding to cache operation and parameter key.
  auto iter = rpc_operators_.find(cache_operation);
  if (iter == rpc_operators_.end()) {
    MS_LOG(ERROR) << "Can not find rpc operator for cache operation: " << cache_operation;
    return false;
  }

  const std::vector<SendRecvPairList> &send_recv_pair_lists = iter->second;
  const SenderPtr &sender = send_recv_pair_lists[server_rank_id][param_key].first;
  MS_ERROR_IF_NULL(sender);

  int64_t ids_num = SizeToLong(keys_len / sizeof(int));
  ShapeVector ids_shape = {ids_num};
  ShapeVector values_shape;
  float fake_value = 0.0;

  if (values == nullptr && values_len == 0) {
    values_shape = {1, 1};
    values = &fake_value;
    values_len = sizeof(fake_value);
  } else {
    MS_EXCEPTION_IF_ZERO("embedding_dim", embedding_dim);
    int64_t embed_vec_num = SizeToLong(values_len / sizeof(float) / embedding_dim);
    if (embed_vec_num != ids_num) {
      MS_LOG(EXCEPTION) << "The embedding vector number[" << embed_vec_num << "] shouled be equal to ids number["
                        << ids_num << "] which will be send to remote.";
    }
    values_shape = {embed_vec_num, SizeToLong(embedding_dim)};
  }

  std::vector<ShapeVector> shapes = {ids_shape, values_shape, {static_cast<int64_t>(1)}};
  std::vector<TypeId> data_types = {kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeInt32};

  int32_t service_id = GetCacheOpsServiceId(cache_operation, param_key);
  AddressPtrList data_list = {std::make_shared<Address>(const_cast<void *>(keys), keys_len),
                              std::make_shared<Address>(const_cast<void *>(values), values_len),
                              std::make_shared<Address>(&service_id, sizeof(int32_t))};

  // Send data.
  return sender->Send(shapes, data_types, data_list, finalize_remote, sync);
}

std::unique_ptr<std::vector<char>> EmbeddingCachePrefetchActor::ReceiveFromRemote(const std::string &cache_operation,
                                                                                  int32_t param_key,
                                                                                  size_t server_rank_id) const {
  // Find receiver corresponding to cache operation and parameter key.
  auto iter = rpc_operators_.find(cache_operation);
  if (iter == rpc_operators_.end()) {
    MS_LOG(ERROR) << "Can not find rpc operator for cache operation: " << cache_operation;
    return nullptr;
  }

  const std::vector<SendRecvPairList> &send_recv_pair_lists = iter->second;
  const ReceiverPtr &receiver = send_recv_pair_lists[server_rank_id][param_key].second;
  MS_EXCEPTION_IF_NULL(receiver);
  // Receive data.
  return receiver->Receive();
}

bool EmbeddingCachePrefetchActor::RetrieveEmbeddings(
  const int *ids, size_t ids_num, const std::vector<std::vector<int>> &slice_ids_list,
  const std::vector<std::unique_ptr<std::vector<char>>> &slice_embeddings_list, std::vector<float> *outputs) const {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(outputs);

  if (ids_num == 0) {
    MS_LOG(WARNING) << "The ids number is 0";
    return true;
  }

  // Merge all slice ids and embedding data address into ids_to_addrs map.
  mindspore::HashMap<int, const float *> ids_to_addrs;
  size_t embedding_dim = outputs->size() / ids_num;
  size_t offset = 0;
  for (size_t i = 0; i < slice_ids_list.size(); i++) {
    const std::vector<int> &slice_ids = slice_ids_list[i];
    if (slice_ids.empty()) {
      continue;
    }
    const std::unique_ptr<std::vector<char>> &slice_embeddings = slice_embeddings_list[i];
    MS_ERROR_IF_NULL(slice_embeddings);
    const float *embeddings_data = reinterpret_cast<float *>(slice_embeddings->data());
    for (size_t j = 0; j < slice_ids.size(); j++) {
      (void)ids_to_addrs.emplace(slice_ids[j], embeddings_data + offset);
      offset += embedding_dim;
    }
    offset = 0;
  }

  float *outputs_data = outputs->data();
  size_t dst_size = embedding_dim * sizeof(float);
  size_t src_size = dst_size;
  offset = 0;

  // Retrieve embeddings by input ids order.
  for (size_t i = 0; i < ids_num; i++) {
    auto id = ids[i];
    auto iter = ids_to_addrs.find(id);
    if (iter == ids_to_addrs.end()) {
      MS_LOG(WARNING) << "Can not find id[" << id << "]";
      continue;
    }

    auto ret = memcpy_s(outputs_data + offset, dst_size, iter->second, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
      return false;
    }
    offset += embedding_dim;
  }
  return true;
}

void EmbeddingCachePrefetchActor::SyncEmbeddingTable() {
  std::lock_guard<std::mutex> locker(sync_embedding_table_mutex_);
  // Do not synchronize in case of abnormally finalizing.
  if (!running_) {
    return;
  }

  if (finish_sync_embedding_table_) {
    return;
  }
  if (!initialized_) {
    return;
  }
  if (!SyncHostEmbeddingTable()) {
    MS_LOG(ERROR) << "SyncHostEmbeddingTable failed.";
  }
  if (!SyncDeviceEmbeddingTable()) {
    MS_LOG(ERROR) << "SyncDeviceEmbeddingTable failed.";
  }
  finish_sync_embedding_table_ = true;
}

bool EmbeddingCachePrefetchActor::SyncHostEmbeddingTable() {
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_->host_hash_map_);
  const auto &hash_id_to_index =
    embedding_cache_table_manager.embedding_host_cache_->host_hash_map_->hash_id_to_index();
  size_t swap_indices_lens = hash_id_to_index.size();
  if (swap_indices_lens == 0) {
    return true;
  }

  std::unique_ptr<int[]> host_to_server_ids_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_ids_ptr);
  std::unique_ptr<int[]> host_to_server_indices_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_indices_ptr);
  size_t idx = 0;
  for (const auto &item : hash_id_to_index) {
    host_to_server_ids_ptr[idx] = item.first;
    host_to_server_indices_ptr[idx++] = item.second;
  }
  for (const auto &item : embedding_cache_table_manager.hash_tables_) {
    const auto &hash_info = item.second;
    std::vector<float> swap_out_data;
    auto embedding_size = hash_info.embedding_size;
    swap_out_data.resize(swap_indices_lens * embedding_size);
    auto host_hash_table_addr = hash_info.host_address.get();
    MS_ERROR_IF_NULL(host_hash_table_addr);
    RETURN_IF_FALSE(LookupLocalHostCache(embedding_size, swap_indices_lens, host_hash_table_addr,
                                         host_to_server_indices_ptr.get(), swap_out_data.data()));

    RETURN_IF_FALSE_WITH_LOG(
      PushEmbeddingsToRemote(hash_info.param_key_, host_to_server_ids_ptr.get(), swap_indices_lens,
                             swap_out_data.data(), swap_out_data.size() * sizeof(float)),
      "Push embeddings to remote failed.");
  }
  return true;
}

bool EmbeddingCachePrefetchActor::SyncDeviceEmbeddingTable() {
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  const auto &device_hash_map = embedding_cache_table_manager.embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  size_t swap_indices_lens = hash_id_to_index.size();
  if (swap_indices_lens == 0) {
    return true;
  }
  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->device_res_manager_);
  std::unique_ptr<int[]> device_to_server_ids_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(device_to_server_ids_ptr);
  std::unique_ptr<int[]> device_to_server_indices_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(device_to_server_indices_ptr);
  size_t idx = 0;
  for (const auto &item : hash_id_to_index) {
    device_to_server_ids_ptr[idx] = item.first;
    device_to_server_indices_ptr[idx++] = item.second;
  }
  for (const auto &item : embedding_cache_table_manager.hash_tables_) {
    const auto &hash_info = item.second;
    std::vector<float> swap_out_data;
    auto embedding_size = hash_info.embedding_size;
    swap_out_data.resize(swap_indices_lens * embedding_size);
    std::unique_ptr<float[]> device_hash_table_addr_tmp =
      std::make_unique<float[]>(device_hash_map->hash_capacity() * embedding_size);
    MS_ERROR_IF_NULL(device_hash_table_addr_tmp);

    auto hash_table_addr = reinterpret_cast<float *>(hash_info.address.addr);
    MS_ERROR_IF_NULL(hash_table_addr);
    auto hash_table_size = hash_info.address.size;
    RETURN_IF_FALSE_WITH_LOG(
      DeviceEmbeddingOperation::MemcpyDeviceToHostAsync(device_hash_table_addr_tmp.get(), hash_table_addr,
                                                        hash_table_size, device_context_, stream_id_),
      "Memcpy device to host asynchronously failed.");
    RETURN_IF_FALSE_WITH_LOG(device_context_->device_res_manager_->SyncStream(stream_id_),
                             "Synchronize stream failed.");
    RETURN_IF_FALSE(LookupLocalHostCache(embedding_size, swap_indices_lens, device_hash_table_addr_tmp.get(),
                                         device_to_server_indices_ptr.get(), swap_out_data.data()));

    RETURN_IF_FALSE_WITH_LOG(
      PushEmbeddingsToRemote(hash_info.param_key_, device_to_server_ids_ptr.get(), swap_indices_lens,
                             swap_out_data.data(), swap_out_data.size() * sizeof(float)),
      "Push embeddings to remote failed.");
  }
  return true;
}

bool EmbeddingCachePrefetchActor::FinalizeRemote() {
  for (size_t i = 0; i < server_num_; i++) {
    size_t embedding_dim = 1;
    int id = 0;
    float value = 0.0;
    RETURN_IF_FALSE_WITH_LOG(SendToRemote(distributed::kLookupEmbeddingCache, 0, i, embedding_dim, &id, sizeof(int),
                                          &value, sizeof(float), true),
                             "Send finalize request to remote failed.");
  }

  return true;
}

std::string EmbeddingCachePrefetchActor::channel_name() {
  std::lock_guard<std::mutex> locker(channel_mutex_);
  return channel_name_;
}

void EmbeddingCachePrefetchActor::set_channel_name(const std::string channel_name) {
  if (channel_name_ == channel_name) {
    return;
  }
  std::lock_guard<std::mutex> locker(channel_mutex_);
  channel_name_ = channel_name;
}

void EmbeddingCachePrefetchActor::WaitDataChannelInit() {
  MS_LOG(INFO) << "Begin wait embedding cache data channel init.";
  auto channel = channel_name();
  if (channel.empty()) {
    std::unique_lock<std::mutex> locker(data_mutex_);
    data_parser_.wait(locker, [this] { return !channel_name_.empty() || running_ == false; });
    if (!running_) {
      return;
    }
  }
  MS_LOG(INFO) << "End wait embedding cache data channel init.";
}

void EmbeddingCachePrefetchActor::WaitInitParametersOnRemote() {
  std::unique_lock<std::mutex> locker(data_mutex_);
  // Note: wait to finish embedding lookup from remote.
  finish_init_parameters_on_remote_ = true;
  data_parser_.notify_one();
}

void EmbeddingCachePrefetchActor::SetErrorInfo(const std::string &error_info) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  error_info_ = error_info;
}

void EmbeddingCachePrefetchActor::BuildRpcOperators() {
  // The cache operation support LookupEmbeddingCache and UpdateEmbeddingCache currently.
  for (const auto &cache_op : distributed::kEmbeddingCacheOps) {
    rpc_operators_[cache_op] = std::vector<SendRecvPairList>();
    rpc_operators_[cache_op].resize(server_num_);
  }

  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  uint32_t worker_rank_id = node->rank_id();

  // Create sender and receiver pairs for different cache operation, server and parameter key.
  for (auto &item : rpc_operators_) {
    const std::string &cache_op = item.first;
    std::vector<SendRecvPairList> &send_recv_pair_lists = item.second;
    for (uint32_t i = 0; i < server_num_; i++) {
      SendRecvPairList &send_recv_pair_list = send_recv_pair_lists[i];
      send_recv_pair_list.resize(embedding_cache_table_manager.hash_tables_.size());

      for (const auto &table : embedding_cache_table_manager.hash_tables_) {
        int32_t key = table.second.param_key_;
        if (key >= SizeToInt(embedding_cache_table_manager.hash_tables_.size()) || key < 0) {
          MS_LOG(EXCEPTION) << "Invalid parameter key: " << key;
        }

        send_recv_pair_list[key] = CreateSenderReceiverPair(worker_rank_id, i, cache_op, key, cpu_device_context_);
      }
    }
  }
}

void EmbeddingCachePrefetchActor::LinkRpcOperators() {
  std::vector<SenderPtr> senders;
  std::vector<ReceiverPtr> receivers;
  for (const auto &item : rpc_operators_) {
    const std::vector<SendRecvPairList> &send_recv_pair_lists = item.second;
    for (const SendRecvPairList &send_recv_pair_list : send_recv_pair_lists) {
      for (const SendRecvPair &pair : send_recv_pair_list) {
        senders.push_back(pair.first);
        receivers.push_back(pair.second);
      }
    }
  }

  // Must start server and register route table before looking up route and connecting.
  // Start servers of receiver and register route table.
  for (auto &receiver : receivers) {
    MS_EXCEPTION_IF_NULL(receiver);
    if (!receiver->StartServer()) {
      MS_LOG(EXCEPTION) << "Failed to start server for the receiver.";
    }
  }

  // Lookup route and connect to servers for sender.
  for (auto &sender : senders) {
    MS_EXCEPTION_IF_NULL(sender);
    if (!sender->ConnectServer()) {
      MS_LOG(EXCEPTION) << "Failed to connect servers for the sender.";
    }
  }
}

bool Sender::Send(const std::vector<ShapeVector> &shapes, const std::vector<TypeId> data_types,
                  const AddressPtrList &data_list, bool finalize_remote, bool sync) const {
  MS_ERROR_IF_NULL(receiver_);
  auto message = BuildRpcMessage(shapes, data_types, data_list, receiver_->get_url(), server_url_, finalize_remote);
  MS_ERROR_IF_NULL(message);
  MS_ERROR_IF_NULL(client_);
  if (sync) {
    return client_->SendSync(std::move(message));
  }

  client_->SendAsync(std::move(message));
  return true;
}

Sender::~Sender() {
  if (client_) {
    try {
      if (!client_->Disconnect(server_url_)) {
        MS_LOG(ERROR) << "Failed to disconnect tcp client.";
      }
      client_->Finalize();
      client_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Failed to disconnect and finalize tcp client, error message: " << e.what();
    }
  }
  receiver_ = nullptr;
}

bool Sender::ConnectServer() {
  client_ = std::make_unique<TCPClient>();
  MS_ERROR_IF_NULL(client_);
  if (!client_->Initialize()) {
    MS_LOG(ERROR) << "Failed to initialize tcp server for send actor.";
    return false;
  }

  // Lookup peer receiver addresses.
  MS_ERROR_IF_NULL(route_table_proxy_);
  auto peer_actor_address = route_table_proxy_->LookupRoute(inter_process_edge_);
  server_url_ = peer_actor_address.ip() + ":" + std::to_string(peer_actor_address.port());

  auto free_callback = std::bind(&Sender::FreeMessage, this, std::placeholders::_1);
  size_t retry_count = 60;

  bool ret = client_->Connect(server_url_, retry_count, free_callback);
  if (!ret) {
    MS_LOG(ERROR) << "Failed to connect to server of edge: " << inter_process_edge_ << ", server_url: " << server_url_;
    return false;
  }

  MS_LOG(INFO) << "Successfully connect to server " << server_url_
               << ", inter process edge name: " << inter_process_edge_;
  return true;
}

std::unique_ptr<MessageBase> Sender::BuildRpcMessage(const std::vector<ShapeVector> &shapes,
                                                     const std::vector<TypeId> data_types,
                                                     const AddressPtrList &data_list, const std::string &from_url,
                                                     const std::string &to_url, bool finalize_remote) const {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_ERROR_IF_NULL_W_RET_VAL(message, nullptr);
  message->from = AID("", from_url);
  message->to = AID("", to_url);

  if (shapes.size() != data_list.size()) {
    MS_LOG(ERROR) << "The shape list size[" << shapes.size() << "] should be equal to data list size["
                  << data_list.size() << "]";
  }

  if (data_types.size() != data_list.size()) {
    MS_LOG(ERROR) << "The date type list size[" << data_types.size() << "] should be equal to data list size["
                  << data_list.size() << "]";
  }

  RpcDataPtr rpc_data = nullptr;
  size_t data_size = CalDataSize(shapes, data_types, data_list, finalize_remote);
  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
  rpc_data = static_cast<RpcDataPtr>(cpu_device_context_->device_res_manager_->AllocateMemory(data_size));
  MS_EXCEPTION_IF_NULL(rpc_data);
  message->data = rpc_data;
  message->size = data_size;

  size_t offset = 0;
  for (size_t i = 0; i < data_list.size(); i++) {
    const ShapeVector &shape = shapes[i];
    const AddressPtr &data = data_list[i];
    const TypeId &type_id = data_types[i];

    rpc::DynamicShapeMessage ds_pb_msg;
    ds_pb_msg.set_type_id(type_id);
    *ds_pb_msg.mutable_shape_vector() = {shape.begin(), shape.end()};
    std::string ds_pb_msg_str = ds_pb_msg.SerializeAsString();

    // Message format:
    // |RPC_DYNAMIC_SHAPE_DATA | dynamic shape PB data size |---dynamic shape PB data----|---real data----|
    // 1. The dynamic shape header.
    if (EOK !=
        memcpy_s(rpc_data + offset, strlen(kRpcDynamicShapeData), kRpcDynamicShapeData, strlen(kRpcDynamicShapeData))) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for kRpcDynamicShapeData";
    }
    offset += strlen(kRpcDynamicShapeData);

    // 2. The size of the protobuf DynamicShapeMessage.
    size_t ds_pb_msg_size = ds_pb_msg_str.size();
    if (EOK != memcpy_s(rpc_data + offset, sizeof(ds_pb_msg_size), &ds_pb_msg_size, sizeof(ds_pb_msg_size))) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for pb message size.";
    }
    offset += sizeof(ds_pb_msg_size);

    // 3. Protobuf DynamicShapeMessage.
    if (EOK != memcpy_s(rpc_data + offset, ds_pb_msg_str.size(), ds_pb_msg_str.c_str(), ds_pb_msg_str.size())) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for pb message.";
    }
    offset += ds_pb_msg_str.size();

    // 4. The real data buffer need to be sent.
    MS_EXCEPTION_IF_NULL(data);
    if (EOK != memcpy_s(rpc_data + offset, data->size, data->addr, data->size)) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for real data.";
    }
    offset += data->size;
  }

  // 5. Finalize remote command.
  if (finalize_remote) {
    size_t header_len = strlen(distributed::kFinalizeMuxRecvActor);
    if (EOK != memcpy_s(rpc_data + offset, header_len, distributed::kFinalizeMuxRecvActor, header_len)) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for kFinalizeMuxRecvActor.";
    }
    offset += header_len;

    if (EOK != memcpy_s(rpc_data + offset, sizeof(finalize_remote), &finalize_remote, sizeof(finalize_remote))) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for finalize_remote.";
    }
  }

  return message;
}

bool Sender::FreeMessage(void *data) {
  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  cpu_device_context_->device_res_manager_->FreeMemory(data);
  return true;
}

size_t Sender::CalDataSize(const std::vector<ShapeVector> &shapes, const std::vector<TypeId> data_types,
                           const AddressPtrList &data_list, bool finalize_remote) const {
  size_t data_size = 0;
  for (size_t i = 0; i < data_list.size(); i++) {
    const ShapeVector &shape = shapes[i];
    const AddressPtr &data = data_list[i];
    const TypeId &type_id = data_types[i];

    rpc::DynamicShapeMessage ds_pb_msg;
    ds_pb_msg.set_type_id(type_id);
    *ds_pb_msg.mutable_shape_vector() = {shape.begin(), shape.end()};
    std::string ds_pb_msg_str = ds_pb_msg.SerializeAsString();
    data_size += strlen(kRpcDynamicShapeData);
    data_size += sizeof(size_t);
    data_size += ds_pb_msg_str.size();
    MS_EXCEPTION_IF_NULL(data);
    data_size += data->size;
  }
  if (finalize_remote) {
    data_size += strlen(distributed::kFinalizeMuxRecvActor);
    data_size += sizeof(finalize_remote);
  }
  return data_size;
}

Receiver::~Receiver() {
  if (server_) {
    try {
      server_->Finalize();
      server_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Failed to finalize tcp server, error message: " << e.what();
    }
  }
  received_buffer_ = nullptr;
}

std::unique_ptr<std::vector<char>> Receiver::Receive() {
  std::unique_lock<std::mutex> locker(received_msg_mtx_);
  // The maximum time(300 seconds) to wait to receive message.
  const int64_t longest_time_to_wait = 300;
  auto ret = received_msg_cv_.wait_for(locker, std::chrono::seconds(longest_time_to_wait),
                                       [this] { return received_msg_.load(); });
  if (!ret) {
    MS_LOG(ERROR) << "Receive message timeout";
    return nullptr;
  }

  std::unique_ptr<std::vector<char>> output = std::move(received_buffer_);
  MS_EXCEPTION_IF_NULL(output);
  received_msg_ = false;
  return output;
}

bool Receiver::StartServer() {
  // 1. Create a tcp server and start listening.
  server_ = std::make_unique<TCPServer>();
  MS_EXCEPTION_IF_NULL(server_);

  std::function<void *(size_t size)> allocate_callback =
    std::bind(&Receiver::AllocateMessage, this, std::placeholders::_1);
  if (!server_->Initialize(allocate_callback)) {
    MS_LOG(EXCEPTION) << "Failed to initialize tcp server for recv actor";
  }
  ip_ = server_->GetIP();
  port_ = server_->GetPort();
  std::string server_url = ip_ + ":" + std::to_string(port_);

  // 2. Set the message handler of the server.
  server_->SetMessageHandler(std::bind(&Receiver::HandleMessage, this, std::placeholders::_1));

  // 3. Register the server address to route table. The server should not be connected before this step is done.
  MS_LOG(INFO) << "Start server for receiver. Server address: " << server_url
               << ", inter process edge name: " << inter_process_edge_;
  distributed::cluster::topology::ActorAddress recv_actor_addresss;
  recv_actor_addresss.set_actor_id(inter_process_edge_);
  recv_actor_addresss.set_ip(ip_);
  recv_actor_addresss.set_port(port_);
  MS_EXCEPTION_IF_NULL(route_table_proxy_);
  if (!route_table_proxy_->RegisterRoute(inter_process_edge_, recv_actor_addresss)) {
    MS_LOG(EXCEPTION) << "Failed to register route for " << inter_process_edge_ << " " << server_url
                      << " when starting server.";
  }
  return true;
}

bool Receiver::ParseDynamicShapeData(const char *msg_body, size_t msg_len,
                                     std::pair<const void *, size_t> *data) const {
  MS_ERROR_IF_NULL(msg_body);
  MS_ERROR_IF_NULL(data);
  // 1. Check whether received data is valid dynamic shape data.
  size_t dynamic_shape_header_size = strlen(kRpcDynamicShapeData);
  if (msg_len <= dynamic_shape_header_size) {
    MS_LOG(ERROR) << "Received data is not dynamic shape, data length: " << msg_len;
    return false;
  }
  std::string msg_dynamic_shape_header(msg_body, dynamic_shape_header_size);
  if (msg_dynamic_shape_header != kRpcDynamicShapeData) {
    MS_LOG(ERROR) << "Received data is not dynamic shape, not find dynamic shape header: " << kRpcDynamicShapeData;
    return false;
  }

  size_t offset = dynamic_shape_header_size;
  // 2. Parse the size of dynamic shape serialized protobuf message.
  if (offset + sizeof(size_t) >= msg_len) {
    MS_LOG(ERROR) << "Received data is incomplete";
    return false;
  }
  size_t dynamic_shape_pb_size = *(reinterpret_cast<const size_t *>(msg_body + offset));
  offset += sizeof(size_t);
  if (offset + dynamic_shape_pb_size >= msg_len) {
    MS_LOG(ERROR) << "The dynamic shape pb data is incomplete";
    return false;
  }

  // 3. Deserialize the dynamic shape serialized protobuf message.
  rpc::DynamicShapeMessage pb_msg;
  (void)pb_msg.ParseFromArray(msg_body + offset, dynamic_shape_pb_size);
  offset += dynamic_shape_pb_size;
  size_t received_data_len = msg_len - offset;

  // 4. The data integrity check.
  ShapeVector shapes(pb_msg.shape_vector().begin(), pb_msg.shape_vector().end());
  TypeId data_type = static_cast<TypeId>(pb_msg.type_id());
  int64_t expected_data_len = 1;
  if (!kernel::GetShapeSize(shapes, TypeIdToType(data_type), &expected_data_len)) {
    MS_LOG(ERROR) << "Getting shape size for shape " << shapes << " failed.";
    return false;
  }
  if (LongToSize(expected_data_len) != received_data_len) {
    MS_LOG(ERROR) << "Received data is incomplete, expected size: " << expected_data_len
                  << ", but received data size: " << received_data_len;
    return false;
  }
  // 5. Get real data addr and size.
  *data = std::make_pair(msg_body + offset, received_data_len);
  return true;
}

MessageBase *Receiver::HandleMessage(MessageBase *const msg) {
  if (msg == nullptr) {
    MS_LOG(WARNING) << "Received message pointer is nullptr";
    return distributed::rpc::NULL_MSG;
  }

  RpcDataPtr data = static_cast<RpcDataPtr>(msg->data);
  size_t data_size = msg->size;
  // The data pair: <addr of data, size of data>.
  std::pair<const void *, size_t> real_data;
  // Get real data addr and size.
  if (!ParseDynamicShapeData(data, data_size, &real_data)) {
    MS_LOG(EXCEPTION) << "Parse dynamic shape data failed.";
  }

  std::unique_lock<std::mutex> locker(received_msg_mtx_);
  received_buffer_ = std::make_unique<std::vector<char>>();
  received_buffer_->resize(real_data.second);
  MS_EXCEPTION_IF_NULL(real_data.first);

  int ret = memcpy_s(received_buffer_->data(), received_buffer_->size(), real_data.first, real_data.second);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy for received data failed, errno[" << ret << "]";
  }

  received_msg_ = true;
  received_msg_cv_.notify_one();

  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
  cpu_device_context_->device_res_manager_->FreeMemory(data);

  delete msg;
  return distributed::rpc::NULL_MSG;
}

void *Receiver::AllocateMessage(size_t size) {
  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
  void *data = cpu_device_context_->device_res_manager_->AllocateMemory(size);
  MS_EXCEPTION_IF_NULL(data);
  return data;
}
}  // namespace runtime
}  // namespace mindspore
