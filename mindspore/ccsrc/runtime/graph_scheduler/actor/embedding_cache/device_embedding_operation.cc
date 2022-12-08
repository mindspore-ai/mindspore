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

#include "runtime/graph_scheduler/actor/embedding_cache/device_embedding_operation.h"

namespace mindspore {
namespace runtime {
bool DeviceEmbeddingOperation::ParseHostDataHostToDevice(int id, size_t data_step, size_t graph_running_step,
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

bool DeviceEmbeddingOperation::ParseHostDataDeviceToHost(size_t data_step, size_t graph_running_step,
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
}  // namespace runtime
}  // namespace mindspore
