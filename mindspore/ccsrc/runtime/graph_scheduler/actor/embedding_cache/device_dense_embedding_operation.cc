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

#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "runtime/graph_scheduler/actor/embedding_cache/device_dense_embedding_operation.h"

namespace mindspore {
namespace runtime {
const size_t kInputIndexZero = 0;
const size_t kInputIndexOne = 1;
const size_t kInputIndexTwo = 2;
using distributed::INVALID_INDEX_VALUE;
using kernel::Address;
using kernel::AddressPtrList;

void DeviceDenseEmbeddingOperation::LookupEmbeddingTable(size_t indices_num, size_t embedding_size,
                                                         size_t first_dim_size, const float *input_addr,
                                                         const int *indices_addr, float *output_addr) {
  MS_ERROR_IF_NULL_WO_RET_VAL(input_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(indices_addr);
  MS_ERROR_IF_NULL_WO_RET_VAL(output_addr);

  auto type_size = sizeof(float);
  size_t lens = embedding_size * type_size;
  for (size_t i = 0; i < indices_num; ++i) {
    int index = indices_addr[i];
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = index * embedding_size;
      auto ret = memcpy_s(output_addr, (indices_num - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memcpy failed, errno[" << ret << "]";
        *running_ = false;
        return;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_num - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
        *running_ = false;
        return;
      }
    }
    output_addr += embedding_size;
  }
}

bool DeviceDenseEmbeddingOperation::CountCacheMissIds(int *batch_ids, const size_t batch_ids_num, size_t data_step,
                                                      size_t graph_running_step, bool *device_cache_need_wait_graph,
                                                      bool *host_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(batch_ids);

  std::unique_ptr<int[]> hash_index = std::make_unique<int[]>(batch_ids_num);
  MS_ERROR_IF_NULL(hash_index);

  statistics_info_->batch_id_count_ = batch_ids_num;
  std::unique_ptr<bool[]> in_device = std::make_unique<bool[]>(batch_ids_num);
  std::unique_ptr<bool[]> out_range = std::make_unique<bool[]>(batch_ids_num);
  auto ret = memset_s(in_device.get(), batch_ids_num * sizeof(bool), 0, batch_ids_num * sizeof(bool));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
    return false;
  }
  ret = memset_s(out_range.get(), batch_ids_num * sizeof(bool), 0, batch_ids_num * sizeof(bool));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
    return false;
  }

  // 1. Analyze the hit/miss info of the local host cache and device cache.
  RETURN_IF_FALSE_WITH_LOG(
    CheckCacheHitOrOutRange(batch_ids, batch_ids_num, hash_index.get(), in_device.get(), out_range.get(), data_step),
    "Check cache hit or out range failed.");
  RETURN_IF_FALSE_WITH_LOG(actor_->ResetEmbeddingHashMap(), "Reset embedding hash map failed.");

  // 2.calculate the swapping and mapping(feature id to cache index) information of the missing feature id that needs to
  // be inserted into the cache.
  for (size_t i = 0; i < batch_ids_num; i++) {
    if (in_device[i] || out_range[i]) {
      continue;
    }
    bool need_swap_host_to_device = true;
    bool need_swap_device_to_host = true;
    int index = INVALID_INDEX_VALUE;
    RETURN_IF_FALSE_WITH_LOG(ParseDeviceData(batch_ids[i], &need_swap_device_to_host, &need_swap_host_to_device, &index,
                                             data_step, graph_running_step, device_cache_need_wait_graph),
                             "Parse device cache data failed.");
    hash_index[i] = index + local_device_cache_bounds_.first;
    if (need_swap_host_to_device) {
      RETURN_IF_FALSE_WITH_LOG(
        ParseHostDataHostToDevice(batch_ids[i], data_step, graph_running_step, host_cache_need_wait_graph),
        "Parse local host cache data(swap local host cache to device) failed.");
    }
    if (need_swap_device_to_host) {
      RETURN_IF_FALSE_WITH_LOG(ParseHostDataDeviceToHost(data_step, graph_running_step, host_cache_need_wait_graph),
                               "Parse local host cache data(swap device cache to local host) failed.");
    }
  }

  // 3. Replace the batch_ids by hash index for GetNext operator to get hash index as input.
  size_t data_size = batch_ids_num * sizeof(int);
  size_t dest_len = data_size;
  void *data = reinterpret_cast<void *>(batch_ids);
  ret = memcpy_s(data, dest_len, hash_index.get(), data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy hash index failed, errno[" << ret << "]";
    return false;
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_->host_to_device_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);

  auto host_cache_host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  auto device_cache_host_to_device_index = embedding_device_cache_->host_to_device_index.get();
  MS_ERROR_IF_NULL(host_cache_host_to_device_index);
  MS_ERROR_IF_NULL(device_cache_host_to_device_index);

  auto embedding_size = hash_info.embedding_size;
  MS_ERROR_IF_NULL(hash_info.device_address.addr);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  MS_ERROR_IF_NULL(hash_info.host_address);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);
  RETURN_IF_FALSE_WITH_LOG(actor_->LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                        host_cache_host_to_device_index, swap_out_data.get()),
                           "Lookup local host cache failed.");

  RETURN_IF_FALSE_WITH_LOG(
    MemcpyHostToDeviceAsync(embedding_device_cache_->hash_swap_value_addr_, swap_out_data.get(),
                            swap_indices_size * embedding_size * sizeof(float), device_context_, stream_id_),
    "Memcpy host to device asynchronously failed.");
  RETURN_IF_FALSE_WITH_LOG(
    MemcpyHostToDeviceAsync(embedding_device_cache_->hash_swap_index_addr_, device_cache_host_to_device_index,
                            swap_indices_size * sizeof(int), device_context_, stream_id_),
    "Memcpy host to device asynchronously failed.");

  RETURN_IF_FALSE_WITH_LOG(
    UpdateDeviceCache(embedding_device_cache_->hash_swap_index_addr_, embedding_device_cache_->hash_swap_value_addr_,
                      swap_indices_size, cache_vocab_size, embedding_size, hash_table_addr),
    "Update device embedding cache failed.");
  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->device_res_manager_);
  RETURN_IF_FALSE_WITH_LOG(device_context_->device_res_manager_->SyncStream(stream_id_), "Synchronize stream failed.");
  return true;
}

bool DeviceDenseEmbeddingOperation::PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_->device_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);

  auto device_cache_device_to_host_index = embedding_device_cache_->device_to_host_index.get();
  auto host_cache_device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_cache_device_to_host_index);
  MS_ERROR_IF_NULL(host_cache_device_to_host_index);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto embedding_size = hash_info.embedding_size;
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);

  RETURN_IF_FALSE_WITH_LOG(
    MemcpyHostToDeviceAsync(embedding_device_cache_->hash_swap_index_addr_, device_cache_device_to_host_index,
                            swap_indices_size * sizeof(int), device_context_, stream_id_),
    "Memcpy host to device asynchronously failed.");

  RETURN_IF_FALSE_WITH_LOG(
    LookupDeviceCache(embedding_device_cache_->hash_swap_index_addr_, hash_table_addr, swap_indices_size,
                      cache_vocab_size, embedding_size, embedding_device_cache_->hash_swap_value_addr_),
    "Lookup device cache failed.");

  RETURN_IF_FALSE_WITH_LOG(
    MemcpyDeviceToHostAsync(swap_out_data.get(), embedding_device_cache_->hash_swap_value_addr_,
                            swap_indices_size * embedding_size * sizeof(float), device_context_, stream_id_),
    "Memcpy device to host asynchronously failed.");

  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->device_res_manager_);
  RETURN_IF_FALSE_WITH_LOG(device_context_->device_res_manager_->SyncStream(stream_id_), "Synchronize stream failed.");
  RETURN_IF_FALSE_WITH_LOG(
    actor_->InsertLocalHostCache(embedding_size, IntToSize(swap_indices_size), host_cache_device_to_host_index,
                                 swap_out_data.get(), host_hash_table_addr),
    "Insert local host cache failed.");
  return true;
}

bool DeviceDenseEmbeddingOperation::LookupDeviceCache(void *indices, void *embedding_cache, size_t indices_num,
                                                      size_t cache_size, size_t embedding_size, void *outputs) {
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(embedding_cache_lookup_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, kInputIndexZero);
  MS_ERROR_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, kInputIndexOne);
  MS_ERROR_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  auto input_offset_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, kInputIndexTwo);
  MS_ERROR_IF_NULL(input_offset_node);
  auto offset_address = AnfAlgo::GetMutableOutputAddr(input_offset_node, 0);
  MS_ERROR_IF_NULL(offset_address);

  // 2. Infer shape for embedding cache look up kernel(operator name: 'Gather') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_lookup_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_lookup_node_->fullname_with_scope();
    return false;
  }

  // 3. Do embedding cache look up on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int)),
    std::make_shared<Address>(offset_address->GetMutablePtr(), offset_address->GetSize())};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(outputs, indices_num * embedding_size * sizeof(float))};

  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->kernel_executor_);
  auto ret = device_context_->kernel_executor_->LaunchKernel(embedding_cache_lookup_node_, kernel_inputs, {},
                                                             kernel_outputs, stream_id_);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_lookup_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::UpdateDeviceCache(void *indices, void *update_value, size_t indices_num,
                                                      size_t cache_size, size_t embedding_size, void *embedding_cache) {
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(update_value);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(embedding_cache_update_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, kInputIndexZero);
  MS_ERROR_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, kInputIndexOne);
  MS_ERROR_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  auto update_values_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, kInputIndexTwo);
  MS_ERROR_IF_NULL(update_values_node);
  const ShapeVector update_values_shape = {SizeToLong(indices_num), SizeToLong(embedding_size)};
  auto update_values_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, update_values_shape);
  update_values_node->set_abstract(update_values_abstract);

  // 2. Infer shape for embedding cache update kernel(operator name: 'ScatterUpdate') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_update_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_update_node_->fullname_with_scope();
    return false;
  }

  // 3. Do update cache on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int)),
    std::make_shared<Address>(update_value, indices_num * embedding_size * sizeof(float))};
  AddressPtrList kernel_outputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float))};

  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->kernel_executor_);
  auto ret = device_context_->kernel_executor_->LaunchKernel(embedding_cache_update_node_, kernel_inputs, {},
                                                             kernel_outputs, stream_id_);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_update_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::CheckCacheHitOrOutRange(const int *batch_ids, const size_t batch_ids_num,
                                                            int *hash_index, bool *in_device, bool *out_range,
                                                            size_t data_step) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(out_range);

  size_t thread_num = batch_ids_num / kMaxIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t hash_hit_count[kMaxThreadNum] = {0};
  size_t i = 0;
  size_t offset = 0;

  for (; i < thread_num; ++i) {
    if (offset >= batch_ids_num) {
      break;
    }
    size_t proc_len = batch_ids_num / thread_num + (i < (batch_ids_num % thread_num) ? 1 : 0);
    threads[i] =
      std::thread(&DeviceDenseEmbeddingOperation::CheckCacheHitOrOutRangeFunc, this, batch_ids + offset, proc_len,
                  hash_index + offset, in_device + offset, out_range + offset, hash_hit_count + i, data_step);
    offset += proc_len;
  }
  if (offset != batch_ids_num) {
    MS_LOG(WARNING) << "Check id in device inadequate, total:" << batch_ids_num << " checked:" << offset;
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  for (size_t j = 0; j < i; j++) {
    statistics_info_->hash_hit_count_ += hash_hit_count[j];
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::CheckCacheHitOrOutRangeFunc(const int *batch_ids, const size_t batch_ids_num,
                                                                int *hash_index, bool *in_device, bool *out_range,
                                                                size_t *hash_hit_count, size_t data_step) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(out_range);
  MS_ERROR_IF_NULL(hash_hit_count);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();

  for (size_t i = 0; i < batch_ids_num; ++i) {
    if (batch_ids[i] < local_embedding_slice_bounds_.first) {
      hash_index[i] = batch_ids[i] - local_embedding_slice_bounds_.first + local_device_cache_bounds_.first;
      out_range[i] = true;
      continue;
    }
    if (batch_ids[i] >= local_embedding_slice_bounds_.second) {
      hash_index[i] = batch_ids[i] + local_device_cache_bounds_.second;
      out_range[i] = true;
      continue;
    }
    auto iter = hash_id_to_index.find(batch_ids[i]);
    if (iter != hash_id_to_index.end()) {
      hash_index[i] = iter->second + local_device_cache_bounds_.first;
      if (device_hash_map->hash_step(iter->second) != data_step) {
        ++(*hash_hit_count);
        device_hash_map->set_hash_step(iter->second, data_step);
      }
      in_device[i] = true;
    }
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::ParseDeviceData(int id, bool *need_swap_device_to_host,
                                                    bool *need_swap_host_to_device, int *hash_index, size_t data_step,
                                                    size_t graph_running_step, bool *device_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(need_swap_device_to_host);
  MS_ERROR_IF_NULL(need_swap_host_to_device);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);

  int index = INVALID_INDEX_VALUE;
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    *need_swap_device_to_host = false;
    *need_swap_host_to_device = false;
    index = iter->second;
    if (device_hash_map->hash_step(index) != data_step) {
      statistics_info_->hash_hit_count_++;
      device_hash_map->set_hash_step(index, data_step);
    }
  } else {
    int *device_to_host_index = embedding_device_cache_->device_to_host_index.get();
    int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
    int *host_to_device_index = embedding_device_cache_->host_to_device_index.get();
    int *host_to_device_ids = embedding_device_cache_->host_to_device_ids.get();
    MS_ERROR_IF_NULL(host_to_device_index);
    MS_ERROR_IF_NULL(host_to_device_ids);
    auto tmp_device_to_host_size = statistics_info_->device_to_host_size_;
    while (true) {
      // Calculate the mapping of id to index.
      index = device_hash_map->ParseData(id, device_to_host_index, device_to_host_ids, data_step, graph_running_step,
                                         &statistics_info_->device_to_host_size_, device_cache_need_wait_graph);
      if (index == INVALID_INDEX_VALUE) {
        if (!actor_->WaitGraphRun()) {
          return false;
        }
        continue;
      }
      host_to_device_index[statistics_info_->host_to_device_size_] = index;
      host_to_device_ids[statistics_info_->host_to_device_size_] = id;
      statistics_info_->host_to_device_size_++;
      *need_swap_device_to_host = statistics_info_->device_to_host_size_ > tmp_device_to_host_size;
      break;
    }
  }

  *hash_index = index;
  return true;
}

bool DeviceDenseEmbeddingOperation::ParseHostDataHostToDevice(int id, size_t data_step, size_t graph_running_step,
                                                              bool *host_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  int *host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  MS_ERROR_IF_NULL(host_to_device_index);
  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);

  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step) {
      host_hash_map->set_hash_step(index, data_step);
    }
    host_to_device_index[statistics_info_->host_to_device_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    while (true) {
      // Calculate the mapping of id to index.
      auto index = host_hash_map->ParseData(id, host_to_server_index, host_to_server_ids, data_step, graph_running_step,
                                            &(statistics_info_->host_to_server_size_), host_cache_need_wait_graph);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE_WITH_LOG(actor_->WaitGraphRun(), "Wait graph run failed.");
        continue;
      }
      host_to_device_index[statistics_info_->host_to_device_size_ - 1] = index;

      // This feature id has never been seen before, so it's value is initialized using the local random generator.
      if (initialized_ids_.find(id) == initialized_ids_.end()) {
        int *new_id_index = embedding_host_cache_->new_id_index.get();
        MS_ERROR_IF_NULL(new_id_index);
        new_id_index[statistics_info_->new_id_size_] = index;
        initialized_ids_.insert(id);
        // This feature id has been initialized already, so it's latest value has been kept in the remote server.
      } else {
        int *server_to_host_index = embedding_host_cache_->server_to_host_index.get();
        int *server_to_host_ids = embedding_host_cache_->server_to_host_ids.get();
        MS_ERROR_IF_NULL(server_to_host_index);
        MS_ERROR_IF_NULL(server_to_host_ids);
        server_to_host_index[statistics_info_->server_to_host_size_] = index;
        server_to_host_ids[statistics_info_->server_to_host_size_++] = id;
      }
      break;
    }
  }

  return true;
}

bool DeviceDenseEmbeddingOperation::ParseHostDataDeviceToHost(size_t data_step, size_t graph_running_step,
                                                              bool *host_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
  int *device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_to_host_ids);
  MS_ERROR_IF_NULL(device_to_host_index);

  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  int swap_device_to_host_id = device_to_host_ids[statistics_info_->device_to_host_size_ - 1];
  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(swap_device_to_host_id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step) {
      host_hash_map->set_hash_step(index, data_step);
    }
    device_to_host_index[statistics_info_->device_to_host_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    while (true) {
      // Calculate the mapping of id to index.
      auto index = host_hash_map->ParseData(swap_device_to_host_id, host_to_server_index, host_to_server_ids, data_step,
                                            graph_running_step, &statistics_info_->host_to_server_size_,
                                            host_cache_need_wait_graph);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE_WITH_LOG(actor_->WaitGraphRun(), "Wait graph run");
        continue;
      }
      device_to_host_index[statistics_info_->device_to_host_size_ - 1] = index;
      break;
    }
  }

  return true;
}

bool DeviceDenseEmbeddingOperation::InferOpShape(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  opt::dynamic_shape::InferOp(kernel);
  auto args = kernel::GetArgsFromCNode(kernel);
  MS_EXCEPTION_IF_NULL(args);

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (kernel::KRET_OK != kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map)) {
    MS_LOG(ERROR) << "Kernel " << kernel->fullname_with_scope() << " resize failed.";
    return false;
  }
  return true;
}

bool DeviceDenseEmbeddingOperation::MemcpyHostToDeviceAsync(void *dst, const void *src, size_t size,
                                                            const DeviceContext *device_context, size_t stream_id) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  MS_ERROR_IF_NULL(device_context);
  MS_ERROR_IF_NULL(device_context->device_res_manager_);

  void *device_ptr = dst;
  const void *host_ptr = src;

  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(device_ptr, size, kOpFormat_DEFAULT,
                                                                                 kTypeUnknown, ShapeVector());
  MS_ERROR_IF_NULL(device_address);
  RETURN_IF_FALSE_WITH_LOG(device_address->AsyncHostToDevice({}, size, kTypeUnknown, host_ptr, stream_id),
                           "Async memcpy host to device failed.");

  return true;
}

bool DeviceDenseEmbeddingOperation::MemcpyDeviceToHostAsync(void *dst, const void *src, size_t size,
                                                            const DeviceContext *device_context, size_t stream_id) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  MS_ERROR_IF_NULL(device_context);
  MS_ERROR_IF_NULL(device_context->device_res_manager_);

  void *device_ptr = const_cast<void *>(src);
  void *host_ptr = dst;

  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(device_ptr, size, kOpFormat_DEFAULT,
                                                                                 kTypeUnknown, ShapeVector());
  MS_ERROR_IF_NULL(device_address);
  RETURN_IF_FALSE_WITH_LOG(device_address->AsyncDeviceToHost({}, size, kTypeUnknown, host_ptr, stream_id),
                           "Async memcpy device to host failed.");

  return true;
}
}  // namespace runtime
}  // namespace mindspore
