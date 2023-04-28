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
#include "include/backend/distributed/embedding_cache/data_queue_manager.h"

namespace mindspore {
namespace runtime {
using distributed::IdDataInfo;
using distributed::IndexDataInfo;

using distributed::DataQueueManager;
using distributed::cluster::ClusterContext;
using mindspore::session::KernelGraph;
constexpr size_t kDefaultQueueCapacity = 128;

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

// Parallelly generate fixed or random numbers continuously using specified algorithm.
template <typename T, typename Generator, typename Distribution, typename... Args>
void GenerateDistributionParallel(size_t size, T *output, Args... args) {
  std::thread threads[kMaxThreadNum];
  std::random_device rd;
  const std::uint64_t seed = rd();

  // 1. Compute total thread number need to parallel generate distribution and the size of new numbers that each
  // thread need to generate.
  // Once calculation of the normal distribution may produce two random values, so each thread should be responsible for
  // producing an even number of random numbers, except for the last thread.
  auto [thread_num, size_per_thread] = random::ComputeTaskNumSize(size, kMaxThreadNum);

  // For performance, multi-thread concurrency is not required when the total size is small.
  if (thread_num == 1) {
    random::GenerateRandoms<T, Generator, Distribution, Args...>(seed, 0, output, size, args...);
    return;
  }

  // 2. Parallelly generate specified distribution using specified algorithm.
  // Note that the offset need to be set to 'Generator' to prevent generating same sequence of each thread.
  size_t offset = 0;
  for (size_t i = 0; i < thread_num; ++i) {
    size_t task_len = ((i < (thread_num - 1)) ? size_per_thread : (size - ((thread_num - 1) * size_per_thread)));
    threads[i] = std::thread(&random::GenerateRandoms<T, Generator, Distribution, Args...>, seed, offset,
                             output + offset, task_len, args...);
    offset += task_len;
  }

  for (size_t j = 0; j < thread_num; j++) {
    threads[j].join();
  }
}

void DeduplicateId(UniqueIds *unique_ids) {
  MS_EXCEPTION_IF_NULL(unique_ids);

  constexpr size_t kMaxParallelNum = 32;
  size_t parallel_num = unique_ids->multi_batch_data_.size();
  if (parallel_num > kMaxParallelNum) {
    MS_LOG(EXCEPTION) << "The parallel num: " << parallel_num
                      << " can not be greater than max parallel num: " << kMaxParallelNum;
  }
  std::thread threads[kMaxParallelNum];

  std::vector<mindspore::HashSet<int>> unique_batch_ids_sets(parallel_num);
  auto unique_task = [&](int *origin_batch_ids, size_t proc_len, mindspore::HashSet<int> *unique_set) {
    (void)std::for_each(origin_batch_ids, origin_batch_ids + proc_len,
                        [&unique_set](int id) { (void)unique_set->insert(id); });
  };

  size_t i = 0;
  for (; i < parallel_num; ++i) {
    threads[i] = std::thread(unique_task, reinterpret_cast<int *>(unique_ids->multi_batch_data_.at(i)),
                             unique_ids->multi_batch_size_.at(i), &unique_batch_ids_sets[i]);
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }

  for (size_t k = 1; k < parallel_num; ++k) {
    auto end_iter = unique_batch_ids_sets[k].end();
    for (auto iter = unique_batch_ids_sets[k].begin(); iter != end_iter; ++iter) {
      unique_batch_ids_sets[0].insert(*iter);
    }
  }
  const auto &unique_ids_set = unique_batch_ids_sets.front();
  unique_ids->ids_num_ = unique_ids_set.size();
  unique_ids->ids_ = new int[unique_ids->ids_num_];
  MS_EXCEPTION_IF_NULL(unique_ids->ids_);
  size_t index = 0;
  auto unique_ids_ptr = unique_ids->ids_;
  (void)std::for_each(unique_ids_set.begin(), unique_ids_set.end(), [&](int id) { unique_ids_ptr[index++] = id; });
}

void TransformIdsToIndices(mindspore::HashMap<int, int> *unique_ids_to_indices, size_t batch_ids_num, int *batch_ids) {
  auto change_id_to_index_func = [&](int *batch_ids_ptr, size_t proc_len) {
    for (size_t i = 0; i < proc_len; i++) {
      batch_ids_ptr[i] = (*unique_ids_to_indices)[batch_ids_ptr[i]];
    }
  };

  size_t thread_num = batch_ids_num / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t i = 0;
  size_t offset = 0;

  for (; i < thread_num; ++i) {
    size_t proc_len = batch_ids_num / thread_num + (i < (batch_ids_num % thread_num) ? 1 : 0);
    threads[i] = std::thread(change_id_to_index_func, batch_ids + offset, proc_len);
    offset += proc_len;
  }
  if (offset != batch_ids_num) {
    MS_LOG(WARNING) << "Check id in device inadequate, total:" << batch_ids_num << " checked:" << offset;
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
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
  emb_ops_ = new DeviceDenseEmbeddingOperation(this, device_context_, local_embedding_slice_bounds_,
                                               local_device_cache_bounds_, &statistics_info_, stream_id_);
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

  StopPrefetchCachePipeline();
  for (const auto &item : channel_locks_) {
    const auto &channel_ptr = item.second;
    MS_EXCEPTION_IF_NULL(channel_ptr);
    channel_ptr->TryWakeChannel(true);
  }
  WaitPrefetchCacheFinish();

  PsDataPrefetch::GetInstance().NotifyFinalize();

  if (finalize_remote) {
    (void)FinalizeRemote();
  }

  data_parser_.notify_all();

  if (emb_ops_ != nullptr) {
    delete emb_ops_;
    emb_ops_ = nullptr;
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
    distributed::EmbeddingCacheTableManager::GetInstance().WaitForWarmUpHostCacheComplete();
    MS_LOG(INFO) << "Graph running waiting embedding table init end.";
  }
  graph_step_++;
  if (embedding_cache_table_manager.enable_pipeline()) {
    if (channel_name != channel_name_) {
      set_channel_name(channel_name);
      // Create pipeline tasks for this channel
      StartPrefetchCachePipeline(channel_name);
    }
    const auto &iter = channel_locks_.find(channel_name);
    if (iter == channel_locks_.end()) {
      MS_LOG(EXCEPTION) << "Can not find channel lock for channel: " << channel_name;
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    iter->second->TryWakeChannel();
  } else {
    set_channel_name(channel_name);
    if (!PsDataPrefetch::GetInstance().TryWakeChannel(channel_name)) {
      MS_LOG(EXCEPTION) << "TryWakeChannel failed, channel name: " << channel_name;
    }
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
}

void EmbeddingCachePrefetchActor::CreateChannelLock(const std::string &channel_name) {
  if (channel_locks_.find(channel_name) != channel_locks_.end()) {
    return;
  }
  auto sink_size = DataQueueManager::GetInstance().GetSinkSize(channel_name);
  channel_locks_.emplace(channel_name, std::make_shared<PsDataChannel>(channel_name, sink_size));
}

void EmbeddingCachePrefetchActor::CreateBlockQueue(const std::string &channel_name) {
  auto unique_ids_queue = std::make_shared<BlockingQueue<UniqueIds>>(kDefaultQueueCapacity);
  auto cache_analysis_queue = std::make_shared<BlockingQueue<CacheAnalysis>>(kDefaultQueueCapacity);
  auto ids_and_indices_queue = std::make_shared<BlockingQueue<IdsAndIndices>>(kDefaultQueueCapacity);
  (void)channel_to_queues_.emplace(channel_name,
                                   std::make_tuple(unique_ids_queue, cache_analysis_queue, ids_and_indices_queue));
}

void EmbeddingCachePrefetchActor::StartPrefetchCachePipeline(const std::string &channel_name) {
  MS_LOG(INFO) << "Begin StartPrefetchCachePipeline for channel name: " << channel_name;
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (pipeline_stages_.find(channel_name) != pipeline_stages_.end()) {
    return;
  }

  CreateChannelLock(channel_name);
  CreateBlockQueue(channel_name);

  auto thread_list = std::make_shared<std::vector<std::thread>>(kPipelineStageNum);
  pipeline_stages_.emplace(channel_name, thread_list);

  thread_list->at(kIndex0) = std::thread(&EmbeddingCachePrefetchActor::UniqueIdsTask, this, channel_name);
  thread_list->at(kIndex1) = std::thread(&EmbeddingCachePrefetchActor::AnalyseCacheTask, this, channel_name);
  thread_list->at(kIndex2) = std::thread(&EmbeddingCachePrefetchActor::UpdateCacheTask, this, channel_name);
  thread_list->at(kIndex3) = std::thread(&EmbeddingCachePrefetchActor::TransformIdsToIndicesTask, this, channel_name);
  MS_LOG(INFO) << "End StartPrefetchCachePipeline for channel name: " << channel_name;
}

void EmbeddingCachePrefetchActor::StopPrefetchCachePipeline() {
  MS_LOG(INFO) << "Begin StopPrefetchCachePipeline";
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  running_ = false;
  DataQueueManager::GetInstance().CloseAllQueues();
  for (const auto &item : channel_to_queues_) {
    const BlockingQueueTuple &queues_tuple = item.second;
    const auto &unique_ids_queue = std::get<std::shared_ptr<BlockingQueue<UniqueIds>>>(queues_tuple);
    MS_EXCEPTION_IF_NULL(unique_ids_queue);
    unique_ids_queue->Close();

    const auto &cache_analysis_queue = std::get<std::shared_ptr<BlockingQueue<CacheAnalysis>>>(queues_tuple);
    MS_EXCEPTION_IF_NULL(cache_analysis_queue);
    cache_analysis_queue->Close();

    const auto &ids_and_indices_queue = std::get<std::shared_ptr<BlockingQueue<IdsAndIndices>>>(queues_tuple);
    MS_EXCEPTION_IF_NULL(ids_and_indices_queue);
    ids_and_indices_queue->Close();
  }
  MS_LOG(INFO) << "End StopPrefetchCachePipeline";
}

void EmbeddingCachePrefetchActor::WaitPrefetchCacheFinish() {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  for (auto &item : pipeline_stages_) {
    const std::string &channel_name = item.first;
    MS_LOG(INFO) << "Begin stop pipeline for channel: " << channel_name;
    auto stage_threads = item.second;
    MS_EXCEPTION_IF_NULL(stage_threads);
    for (size_t i = 0; i < kPipelineStageNum; ++i) {
      if (stage_threads->at(i).joinable()) {
        stage_threads->at(i).join();
      }
    }
    MS_LOG(INFO) << "End stop pipeline for channel: " << channel_name;
  }
}

void EmbeddingCachePrefetchActor::UniqueIdsTask(const std::string &channel_name) {
  const auto &iter = channel_locks_.find(channel_name);
  if (iter == channel_locks_.end()) {
    MS_LOG(EXCEPTION) << "Can not find channel lock for channel: " << channel_name;
  }
  auto channel_lock = iter->second;
  MS_EXCEPTION_IF_NULL(channel_lock);

  const auto &id_data_queue = DataQueueManager::GetInstance().GetDataQueue(channel_name).first;
  MS_EXCEPTION_IF_NULL(id_data_queue);

  const auto &queue_iter = channel_to_queues_.find(channel_name);
  if (queue_iter == channel_to_queues_.end()) {
    MS_LOG(EXCEPTION) << "Can not find queue for channel: " << channel_name;
  }
  const auto &unique_ids_queue = std::get<std::shared_ptr<BlockingQueue<UniqueIds>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(unique_ids_queue);

  size_t sink_size = DataQueueManager::GetInstance().GetSinkSize(channel_name);
  size_t multi_batch_counter = 0;
  UniqueIds *unique_ids = nullptr;
  while (running_) {
    IdDataInfo *data = id_data_queue->Pop();
    if (!running_) {
      break;
    }
    MS_EXCEPTION_IF_NULL(data);
    int *batch_ids = reinterpret_cast<int *>(data->data_);
    if (batch_ids) {
      // Lock in first stage to support multi-channel case for real input data.
      channel_lock->TryLockChannel();
      MS_EXCEPTION_IF_CHECK_FAIL(IncreaseStep(), "Increase step failed.");
    }

    if (data->end_of_file_ || data->end_of_epoch_) {
      // Push empty data for epoch or file end flag.
      UniqueIds *empty_unique_ids = new UniqueIds();
      MS_EXCEPTION_IF_NULL(empty_unique_ids);
      empty_unique_ids->end_of_epoch_ = data->end_of_epoch_;
      empty_unique_ids->end_of_file_ = data->end_of_file_;
      unique_ids_queue->Push(empty_unique_ids);
      delete data;
      continue;
    }

    if (unique_ids == nullptr) {
      unique_ids = new UniqueIds();
      MS_EXCEPTION_IF_NULL(unique_ids);
    }

    ++multi_batch_counter;
    if (multi_batch_counter < sink_size &&
        multi_batch_counter % embedding_cache_table_manager.multi_batch_threshold_ != 0) {
      unique_ids->multi_batch_data_.push_back(batch_ids);
      unique_ids->multi_batch_size_.push_back(data->size_ / sizeof(int));
      unique_ids->multi_batch_items_.push_back(data->items_);
      continue;
    }
    unique_ids->multi_batch_data_.push_back(batch_ids);
    unique_ids->multi_batch_size_.push_back(data->size_ / sizeof(int));
    unique_ids->multi_batch_items_.push_back(data->items_);

    if (multi_batch_counter == sink_size) {
      multi_batch_counter = 0;
    }

    // Unique for each batch and store unique ids
    DeduplicateId(unique_ids);
    // Push to next stage pipeline queue.
    unique_ids->data_step_ = data_step_;

    unique_ids_queue->Push(unique_ids);
    unique_ids = nullptr;
    delete data;
  }
}

void EmbeddingCachePrefetchActor::AnalyseCacheTask(const std::string &channel_name) {
  const auto &queue_iter = channel_to_queues_.find(channel_name);
  if (queue_iter == channel_to_queues_.end()) {
    MS_LOG(EXCEPTION) << "Can not find queue for channel: " << channel_name;
  }

  const auto &unique_ids_queue = std::get<std::shared_ptr<BlockingQueue<UniqueIds>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(unique_ids_queue);
  const auto &cache_analysis_queue = std::get<std::shared_ptr<BlockingQueue<CacheAnalysis>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(cache_analysis_queue);

  while (running_) {
    UniqueIds *unique_ids = unique_ids_queue->Pop();
    if (!running_) {
      break;
    }
    MS_EXCEPTION_IF_NULL(unique_ids);
    if (unique_ids->end_of_file_ || unique_ids->end_of_epoch_) {
      // Push empty data for epoch or file end flag.
      CacheAnalysis *cache_analysis = new CacheAnalysis();
      MS_EXCEPTION_IF_NULL(cache_analysis);
      cache_analysis->end_of_epoch_ = unique_ids->end_of_epoch_;
      cache_analysis->end_of_file_ = unique_ids->end_of_file_;
      cache_analysis_queue->Push(cache_analysis);
      delete unique_ids;
      continue;
    }
    size_t unique_ids_num = unique_ids->ids_num_;
    int *indices = new int[unique_ids_num];
    MS_EXCEPTION_IF_NULL(indices);

    EmbeddingDeviceCache *embedding_device_cache = new EmbeddingDeviceCache(unique_ids_num);
    MS_EXCEPTION_IF_NULL(embedding_device_cache);
    EmbeddingHostCache *embedding_host_cache = new EmbeddingHostCache(unique_ids_num);
    MS_EXCEPTION_IF_NULL(embedding_host_cache);
    EmbeddingCacheStatisticsInfo *statistics_info = new EmbeddingCacheStatisticsInfo();
    MS_EXCEPTION_IF_NULL(statistics_info);

    // Analyse cache hit/miss
    if (!emb_ops_->AnalyseCache(unique_ids->ids_, unique_ids_num, unique_ids->data_step_, &graph_step_,
                                &device_cache_need_wait_graph_, &host_cache_need_wait_graph_, indices,
                                embedding_device_cache, embedding_host_cache, statistics_info)) {
      MS_LOG(ERROR) << "Analyse cache failed.";
      StopPrefetchCachePipeline();
      return;
    }

    // Push analyse result to update cache queue
    CacheAnalysis *cache_analysis =
      new CacheAnalysis(embedding_device_cache, embedding_host_cache, statistics_info, unique_ids, indices,
                        unique_ids->end_of_epoch_, unique_ids->end_of_file_);
    MS_EXCEPTION_IF_NULL(cache_analysis);
    cache_analysis_queue->Push(cache_analysis);
  }
}

void EmbeddingCachePrefetchActor::UpdateCacheTask(const std::string &channel_name) {
  const auto &queue_iter = channel_to_queues_.find(channel_name);
  if (queue_iter == channel_to_queues_.end()) {
    MS_LOG(EXCEPTION) << "Can not find queue for channel: " << channel_name;
  }

  const auto &cache_analysis_queue = std::get<std::shared_ptr<BlockingQueue<CacheAnalysis>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(cache_analysis_queue);

  const auto &ids_and_indices_queue = std::get<std::shared_ptr<BlockingQueue<IdsAndIndices>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(ids_and_indices_queue);

  while (running_) {
    CacheAnalysis *cache_analysis = cache_analysis_queue->Pop();
    if (!running_) {
      break;
    }
    MS_EXCEPTION_IF_NULL(cache_analysis);
    if (cache_analysis->end_of_file_ || cache_analysis->end_of_epoch_) {
      // Push empty data for epoch end flag.
      IdsAndIndices *ids_and_indices = new IdsAndIndices();
      MS_EXCEPTION_IF_NULL(ids_and_indices);
      ids_and_indices->end_of_epoch_ = cache_analysis->end_of_epoch_;
      ids_and_indices->end_of_file_ = cache_analysis->end_of_file_;
      ids_and_indices_queue->Push(ids_and_indices);
      delete cache_analysis;
      continue;
    }

    for (const auto &item : embedding_cache_table_manager.hash_tables_) {
      const auto &hash_info = item.second;
      MS_EXCEPTION_IF_CHECK_FAIL(PushCacheFromLocalHostToRemote(hash_info, cache_analysis),
                                 "Push cache from local host to remote failed.");
      MS_EXCEPTION_IF_CHECK_FAIL(emb_ops_->PushCacheFromDeviceToLocalHost(hash_info, cache_analysis),
                                 "Push cache from device to local host failed.");
      MS_EXCEPTION_IF_CHECK_FAIL(InitLocalCacheForNewIds(hash_info, cache_analysis),
                                 "Initialize the local cache values using random generator.");
      MS_EXCEPTION_IF_CHECK_FAIL(PullCacheFromRemoteToLocalHost(hash_info, cache_analysis),
                                 "Pull cache from remote to local host failed.");
      MS_EXCEPTION_IF_CHECK_FAIL(emb_ops_->PullCacheFromLocalHostToDevice(hash_info, cache_analysis),
                                 "Pull cache from local host to device failed.");
    }

    IdsAndIndices *ids_and_indices = new IdsAndIndices(cache_analysis->unique_ids_, cache_analysis->indices_,
                                                       cache_analysis->end_of_epoch_, cache_analysis->end_of_file_);

    ids_and_indices_queue->Push(ids_and_indices);

    delete cache_analysis->embedding_host_cache_;
    delete cache_analysis->embedding_device_cache_;
    delete cache_analysis->statistics_info_;
    delete cache_analysis;
  }
}

void EmbeddingCachePrefetchActor::TransformIdsToIndicesTask(const std::string &channel_name) {
  const auto &queue_iter = channel_to_queues_.find(channel_name);
  if (queue_iter == channel_to_queues_.end()) {
    MS_LOG(EXCEPTION) << "Can not find queue for channel: " << channel_name;
  }
  const auto &ids_and_indices_queue = std::get<std::shared_ptr<BlockingQueue<IdsAndIndices>>>(queue_iter->second);
  MS_EXCEPTION_IF_NULL(ids_and_indices_queue);

  const auto &index_data_queue = DataQueueManager::GetInstance().GetDataQueue(channel_name).second;
  MS_EXCEPTION_IF_NULL(index_data_queue);
  while (running_) {
    IdsAndIndices *ids_and_indices = ids_and_indices_queue->Pop();
    if (!running_) {
      break;
    }
    MS_EXCEPTION_IF_NULL(ids_and_indices);
    // Push empty data for epoch end flag.
    if (ids_and_indices->end_of_file_ || ids_and_indices->end_of_epoch_) {
      IndexDataInfo *indices_info = new IndexDataInfo();
      MS_EXCEPTION_IF_NULL(indices_info);
      indices_info->end_of_epoch_ = ids_and_indices->end_of_epoch_;
      indices_info->end_of_file_ = ids_and_indices->end_of_file_;
      index_data_queue->Push(indices_info);
      delete ids_and_indices;
      continue;
    }

    auto *unique_ids = ids_and_indices->unique_ids_;
    MS_EXCEPTION_IF_NULL(unique_ids);
    auto *unique_ids_ptr = unique_ids->ids_;
    auto unique_ids_num = unique_ids->ids_num_;
    auto *unique_indices_ptr = ids_and_indices->indices_;

    mindspore::HashMap<int, int> unique_ids_to_indices;
    for (size_t i = 0; i < unique_ids_num; i++) {
      (void)unique_ids_to_indices.try_emplace(unique_ids_ptr[i], unique_indices_ptr[i]);
    }

    for (size_t i = 0; i < unique_ids->multi_batch_data_.size(); ++i) {
      if (!embedding_cache_table_manager.is_sparse_format()) {
        TransformIdsToIndices(&unique_ids_to_indices, unique_ids->multi_batch_size_.at(i),
                              reinterpret_cast<int *>(unique_ids->multi_batch_data_.at(i)));
      }
      IndexDataInfo *indices_info =
        new IndexDataInfo(unique_ids->multi_batch_data_.at(i), unique_ids->multi_batch_items_.at(i),
                          ids_and_indices->end_of_epoch_, ids_and_indices->end_of_file_);

      if (unique_ids->multi_batch_data_.at(i) != unique_ids->multi_batch_items_.at(i)->at(0).data_ptr) {
        MS_LOG(EXCEPTION) << "The id data ptr is valid";
      }

      index_data_queue->Push(indices_info);
    }

    delete[] ids_and_indices->unique_ids_->ids_;
    delete ids_and_indices->unique_ids_;
    delete[] ids_and_indices->indices_;
    delete ids_and_indices;
  }
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
  const auto &device_hash_map = embedding_cache_table_manager.device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &host_hash_map = embedding_cache_table_manager.host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  device_hash_map->Reset();
  host_hash_map->Reset();
  device_cache_need_wait_graph_ = false;
  host_cache_need_wait_graph_ = false;
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
        StopPrefetchCachePipeline();
        return;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_num - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
        StopPrefetchCachePipeline();
        return;
      }
    }
    output_addr += embedding_size;
  }
}

bool EmbeddingCachePrefetchActor::PushCacheFromLocalHostToRemote(const HashTableInfo &hash_info,
                                                                 const CacheAnalysis *cache_analysis) {
  MS_ERROR_IF_NULL(cache_analysis);
  auto statistics_info = cache_analysis->statistics_info_;
  auto embedding_host_cache = cache_analysis->embedding_host_cache_;
  MS_ERROR_IF_NULL(statistics_info);
  MS_ERROR_IF_NULL(embedding_host_cache);

  auto swap_indices_size = statistics_info->host_to_server_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  auto host_to_server_ids = embedding_host_cache->host_to_server_ids.get();
  MS_ERROR_IF_NULL(host_to_server_ids);
  auto host_to_server_index = embedding_host_cache->host_to_server_index.get();
  MS_ERROR_IF_NULL(host_to_server_index);

  std::vector<float> swap_out_data;
  auto embedding_size = hash_info.embedding_size;
  swap_out_data.resize(swap_indices_size * embedding_size);
  auto host_hash_table_addr = hash_info.host_address;

  RETURN_IF_FALSE_WITH_LOG(LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                host_to_server_index, swap_out_data.data()),
                           "Lookup local host cache failed.");
  RETURN_IF_FALSE_WITH_LOG(PushEmbeddingsToRemote(hash_info.param_key_, host_to_server_ids, swap_indices_size,
                                                  swap_out_data.data(), swap_out_data.size() * sizeof(float)),
                           "Push embeddings to remote failed.");
  return true;
}

bool EmbeddingCachePrefetchActor::PullCacheFromRemoteToLocalHost(const HashTableInfo &hash_info,
                                                                 const CacheAnalysis *cache_analysis) {
  MS_ERROR_IF_NULL(cache_analysis);
  auto statistics_info = cache_analysis->statistics_info_;
  auto embedding_host_cache = cache_analysis->embedding_host_cache_;
  MS_ERROR_IF_NULL(statistics_info);
  MS_ERROR_IF_NULL(embedding_host_cache);

  auto swap_indices_size = statistics_info->server_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  auto server_to_host_ids = embedding_host_cache->server_to_host_ids.get();
  MS_ERROR_IF_NULL(server_to_host_ids);
  auto server_to_host_index = embedding_host_cache->server_to_host_index.get();
  MS_ERROR_IF_NULL(server_to_host_index);

  auto host_hash_table_addr = hash_info.host_address;
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

bool EmbeddingCachePrefetchActor::InitLocalCacheForNewIds(const HashTableInfo &hash_info,
                                                          const CacheAnalysis *cache_analysis) {
  MS_ERROR_IF_NULL(cache_analysis);
  auto statistics_info = cache_analysis->statistics_info_;
  auto embedding_host_cache = cache_analysis->embedding_host_cache_;
  MS_ERROR_IF_NULL(statistics_info);
  MS_ERROR_IF_NULL(embedding_host_cache);

  auto new_id_size = statistics_info->new_id_size_;
  if (new_id_size == 0) {
    return true;
  }

  auto new_id_index = embedding_host_cache->new_id_index.get();
  MS_ERROR_IF_NULL(new_id_index);

  // Compute the feature values size needed to be initialized.
  auto embedding_size = hash_info.embedding_size;
  auto total_size = new_id_size * embedding_size;
  std::vector<float> init_result(total_size, 0);

  // Initialize accumulate values with the configured constant value.
  if (hash_info.param_init_info_.param_type_ == distributed::ParamType::kAccumulation) {
    auto init_value = hash_info.param_init_info_.init_val_;
    GenerateDistributionParallel<DataType, Generator, ConstantDistribution>(total_size, init_result.data(), init_value);
  } else {
    // Initialize embedding values from local random generator for feature ids that have never been seen before.
    const double mean = 0.0;
    const double sigma = 0.01;
    GenerateDistributionParallel<DataType, Generator, NormalDistribution>(total_size, init_result.data(), mean, sigma);
  }

  // Insert initialized feature values into the local hash cache.
  auto host_hash_table_addr = hash_info.host_address;
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
          StopPrefetchCachePipeline();
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

bool EmbeddingCachePrefetchActor::DoPushEmbeddingsToRemote(int32_t param_key, const int *ids, size_t ids_num,
                                                           const float *embeddings, size_t embeddings_len) {
  MS_LOG(DEBUG) << "Enter DoPushEmbeddingsToRemote - param_key : " << param_key << ", ids : " << ids
                << ", ids_num : " << ids_num << ", embeddings : " << embeddings
                << ", embeddings_len : " << embeddings_len << ".";
  if (ids_num == 0) {
    MS_LOG(ERROR) << "Invalidate ids num : 0.";
    return false;
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
  MS_LOG(DEBUG) << "Exit DoPushEmbeddingsToRemote.";
  return true;
}

bool EmbeddingCachePrefetchActor::PushEmbeddingsToRemote(int32_t param_key, const int *ids, size_t ids_num,
                                                         const float *embeddings, size_t embeddings_len) {
  MS_EXCEPTION_IF_NULL(ids);
  MS_EXCEPTION_IF_NULL(embeddings);
  MS_EXCEPTION_IF_CHECK_FAIL(ids_num != 0, "The ids_num is 0.");

  const auto ids_boundary = ids + ids_num;
  const auto embeddings_boundary = embeddings + embeddings_len / sizeof(float);
  MS_LOG(DEBUG) << "Enter PushEmbeddingsToRemote - param_key : " << param_key << ", ids : " << ids
                << ", ids_num : " << ids_num << ", embeddings : " << embeddings
                << ", embeddings_len : " << embeddings_len << ", ids_boundary : " << ids_boundary
                << ", embeddings_boundary : " << embeddings_boundary << ".";
  const size_t embeddings_num = embeddings_len / sizeof(float);
  const size_t embeddings_dim = embeddings_num / ids_num;
  MS_EXCEPTION_IF_CHECK_FAIL(embeddings_dim != 0, "The embeddings_dim is 0.");
  // Max batch size : 128Mb.
  const size_t max_batch_size = 1 << 27;
  const size_t batch_num = max_batch_size / embeddings_dim;
  MS_EXCEPTION_IF_CHECK_FAIL(batch_num != 0, "The batch_num is 0.");
  size_t batch_size = ids_num / batch_num;
  size_t batch_remainder = ids_num % batch_num;
  if (batch_remainder != 0) {
    batch_size++;
  }
  MS_LOG(DEBUG) << "batch_size : " << batch_size << ", batch_num << : " << batch_num
                << ", batch_remainder : " << batch_remainder << ".";
  for (size_t count = 0; count != batch_size; count++) {
    auto batch_ids = ids + batch_num * count;
    size_t batch_ids_num = (count != batch_size - 1) ? batch_num : batch_remainder;
    auto batch_embeddings = embeddings + batch_num * count * embeddings_dim;
    size_t batch_embeddings_len = batch_ids_num * embeddings_dim * sizeof(float);
    (void)DoPushEmbeddingsToRemote(param_key, batch_ids, batch_ids_num, batch_embeddings, batch_embeddings_len);
  }
  MS_LOG(DEBUG) << "Exit PushEmbeddingsToRemote.";
  return true;
}

bool EmbeddingCachePrefetchActor::PartitionIds(const int *ids, size_t ids_num,
                                               std::vector<std::vector<int>> *slice_ids_list) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(slice_ids_list);

  size_t partition_num = slice_ids_list->size();
  // There is no need to partition ids for one server case.
  if (partition_num == 1) {
    std::vector<int> &slice_ids = slice_ids_list->front();
    slice_ids.resize(ids_num);
    auto ret = memcpy_s(slice_ids.data(), slice_ids.size() * sizeof(int), ids, ids_num * sizeof(int));
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
      return false;
    }
    return true;
  }

  for (size_t i = 0; i < partition_num; i++) {
    int begin = SizeToInt(remote_embedding_slice_bounds_[i].first);
    int end = SizeToInt(remote_embedding_slice_bounds_[i].second);

    std::vector<int> &slice_ids = slice_ids_list->at(i);
    (void)std::for_each(ids, ids + ids_num, [&](int id) {
      if (id >= begin && id <= end) {
        slice_ids.push_back(id);
      }
    });
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

  size_t partition_num = slice_ids_list->size();
  // There is no need to partition ids and embeddings for one server case.
  if (partition_num == 1) {
    std::vector<int> &slice_ids = slice_ids_list->front();
    std::vector<float> &slice_embeddings = slice_embeddings_list->front();
    slice_ids.resize(ids_num);
    slice_embeddings.resize(embeddings_len / sizeof(float));
    auto ret = memcpy_s(slice_ids.data(), slice_ids.size() * sizeof(int), ids, ids_num * sizeof(int));
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
      return false;
    }
    ret = memcpy_s(slice_embeddings.data(), slice_embeddings.size() * sizeof(float), embeddings, embeddings_len);
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
      return false;
    }
    return true;
  }

  size_t embedding_dim = (embeddings_len / ids_num) / sizeof(float);
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
  MS_ERROR_IF_NULL(embedding_cache_table_manager.host_hash_map_);
  const auto &ids_indices_pairs = embedding_cache_table_manager.host_hash_map_->Export();
  size_t swap_indices_lens = ids_indices_pairs.size();
  if (swap_indices_lens == 0) {
    return true;
  }

  std::unique_ptr<int[]> host_to_server_ids_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_ids_ptr);
  std::unique_ptr<int[]> host_to_server_indices_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_indices_ptr);
  size_t idx = 0;
  MS_EXCEPTION_IF_NULL(emb_ops_);
  const auto &modified_ids = emb_ops_->modified_ids();
  for (const auto &item : ids_indices_pairs) {
    if (modified_ids.find(item.first) != modified_ids.end()) {
      host_to_server_ids_ptr[idx] = item.first;
      host_to_server_indices_ptr[idx++] = item.second;
    }
  }
  swap_indices_lens = idx;
  if (swap_indices_lens == 0) {
    return true;
  }
  for (const auto &item : embedding_cache_table_manager.hash_tables_) {
    const auto &hash_info = item.second;
    std::vector<float> swap_out_data;
    auto embedding_size = hash_info.embedding_size;
    swap_out_data.resize(swap_indices_lens * embedding_size);
    auto host_hash_table_addr = hash_info.host_address;
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
  const auto &device_hash_map = embedding_cache_table_manager.device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &ids_indices_pairs = device_hash_map->Export();
  size_t swap_indices_lens = ids_indices_pairs.size();
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
  for (const auto &item : ids_indices_pairs) {
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

const std::string &EmbeddingCachePrefetchActor::channel_name() {
  std::lock_guard<std::mutex> locker(channel_mutex_);
  return channel_name_;
}

void EmbeddingCachePrefetchActor::set_channel_name(const std::string &channel_name) {
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

  errno_t ret;
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
    if ((ret = memcpy_s(rpc_data + offset, strlen(kRpcDynamicShapeData), kRpcDynamicShapeData,
                        strlen(kRpcDynamicShapeData))) != EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for kRpcDynamicShapeData, errno[" << ret << "].";
    }
    offset += strlen(kRpcDynamicShapeData);

    // 2. The size of the protobuf DynamicShapeMessage.
    size_t ds_pb_msg_size = ds_pb_msg_str.size();
    if ((ret = memcpy_s(rpc_data + offset, sizeof(ds_pb_msg_size), &ds_pb_msg_size, sizeof(ds_pb_msg_size))) != EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for pb message size, errno[" << ret << "].";
    }
    offset += sizeof(ds_pb_msg_size);

    // 3. Protobuf DynamicShapeMessage.
    if ((ret = memcpy_s(rpc_data + offset, ds_pb_msg_str.size(), ds_pb_msg_str.c_str(), ds_pb_msg_str.size())) != EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for pb message, errno[" << ret << "].";
    }
    offset += ds_pb_msg_str.size();

    // 4. The real data buffer need to be sent.
    MS_EXCEPTION_IF_NULL(data);
    if ((ret = memcpy_s(rpc_data + offset, data->size, data->addr, data->size)) != EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for real data, errno[" << ret << "].";
    }
    offset += data->size;
  }

  // 5. Finalize remote command.
  if (finalize_remote) {
    size_t header_len = strlen(distributed::kFinalizeMuxRecvActor);
    if ((ret = memcpy_s(rpc_data + offset, header_len, distributed::kFinalizeMuxRecvActor, header_len)) != EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for kFinalizeMuxRecvActor, errno[" << ret << "].";
    }
    offset += header_len;

    if ((ret = memcpy_s(rpc_data + offset, sizeof(finalize_remote), &finalize_remote, sizeof(finalize_remote))) !=
        EOK) {
      MS_LOG(EXCEPTION) << "Failed to memcpy_s for finalize_remote, errno[" << ret << "].";
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
