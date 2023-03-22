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

#include <vector>
#include <string>
#include <limits>
#include "runtime/graph_scheduler/actor/embedding_cache/device_sparse_embedding_operation.h"
#include "ir/map_tensor.h"

namespace mindspore {
namespace runtime {
bool DeviceSparseEmbeddingOperation::Initialize() {
  DeviceEmbeddingOperation::Initialize();
  BuildEmbeddingCacheEraseKernel();
  return true;
}

bool DeviceSparseEmbeddingOperation::CountCacheMissIds(int *batch_ids, const size_t batch_ids_num, size_t data_step,
                                                       size_t graph_running_step, bool *device_cache_need_wait_graph,
                                                       bool *host_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(batch_ids);

  statistics_info_->batch_id_count_ = batch_ids_num;
  std::unique_ptr<bool[]> in_device = std::make_unique<bool[]>(batch_ids_num);
  auto ret = memset_s(in_device.get(), batch_ids_num * sizeof(bool), 0, batch_ids_num * sizeof(bool));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memset failed, errno[" << ret << "]";
    return false;
  }

  // 1. Analyze the hit/miss info of the local host cache and device cache.
  RETURN_IF_FALSE_WITH_LOG(CheckCacheHit(batch_ids, batch_ids_num, in_device.get(), data_step),
                           "Check cache hit or out range failed.");
  RETURN_IF_FALSE_WITH_LOG(actor_->ResetEmbeddingHashMap(), "Reset embedding hash map failed.");

  // 2.calculate the swapping and mapping(feature id to cache index) information of the missing feature id that needs to
  // be inserted into the cache.
  for (size_t i = 0; i < batch_ids_num; i++) {
    if (in_device[i]) {
      continue;
    }
    bool need_swap_host_to_device = true;
    bool need_swap_device_to_host = true;
    RETURN_IF_FALSE_WITH_LOG(ParseDeviceData(batch_ids[i], &need_swap_device_to_host, &need_swap_host_to_device,
                                             data_step, graph_running_step, device_cache_need_wait_graph),
                             "Parse device cache data failed.");

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

  return true;
}

bool DeviceSparseEmbeddingOperation::PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_->device_to_host_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);

  auto device_cache_device_to_host_ids =
    embedding_cache_table_manager.embedding_device_cache_->device_to_host_ids.get();
  auto host_cache_device_to_host_index =
    embedding_cache_table_manager.embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_cache_device_to_host_ids);
  MS_ERROR_IF_NULL(host_cache_device_to_host_index);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto embedding_size = hash_info.embedding_size;
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);

  // Copy origin id to temp buffer of indices.
  int *tmp_swap_ids = embedding_cache_table_manager.embedding_device_cache_->hash_swap_index_addr_;
  RETURN_IF_FALSE_WITH_LOG(MemcpyHostToDeviceAsync(tmp_swap_ids, device_cache_device_to_host_ids,
                                                   swap_indices_size * sizeof(int), device_context_, stream_id_),
                           "Memcpy host to device asynchronously failed.");

  RETURN_IF_FALSE_WITH_LOG(
    LookupDeviceCache(hash_info.device_address, tmp_swap_ids, hash_table_addr, swap_indices_size, cache_vocab_size,
                      embedding_size, embedding_cache_table_manager.embedding_device_cache_->hash_swap_value_addr_),
    "Lookup device cache failed.");

  // Erase swap out id from device hash table.
  RETURN_IF_FALSE_WITH_LOG(EraseDeviceCache(tmp_swap_ids, swap_indices_size, hash_table_addr, hash_info.device_address),
                           "Erase device cache failed");

  RETURN_IF_FALSE_WITH_LOG(
    MemcpyDeviceToHostAsync(swap_out_data.get(),
                            embedding_cache_table_manager.embedding_device_cache_->hash_swap_value_addr_,
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

bool DeviceSparseEmbeddingOperation::PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) {
  auto swap_indices_size = statistics_info_->host_to_device_size_;
  if (swap_indices_size == 0) {
    return true;
  }

  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_host_cache_);

  auto host_cache_host_to_device_index =
    embedding_cache_table_manager.embedding_host_cache_->host_to_device_index.get();
  auto device_cache_host_to_device_ids =
    embedding_cache_table_manager.embedding_device_cache_->host_to_device_ids.get();
  MS_ERROR_IF_NULL(host_cache_host_to_device_index);
  MS_ERROR_IF_NULL(device_cache_host_to_device_ids);

  auto embedding_size = hash_info.embedding_size;
  MS_ERROR_IF_NULL(hash_info.address.addr);
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  MS_ERROR_IF_NULL(hash_info.host_address);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);
  RETURN_IF_FALSE_WITH_LOG(actor_->LookupLocalHostCache(embedding_size, swap_indices_size, host_hash_table_addr,
                                                        host_cache_host_to_device_index, swap_out_data.get()),
                           "Lookup local host cache failed.");

  RETURN_IF_FALSE_WITH_LOG(
    MemcpyHostToDeviceAsync(embedding_cache_table_manager.embedding_device_cache_->hash_swap_value_addr_,
                            swap_out_data.get(), swap_indices_size * embedding_size * sizeof(float), device_context_,
                            stream_id_),
    "Memcpy host to device asynchronously failed.");
  // Copy origin id to temp buffer of indices.
  RETURN_IF_FALSE_WITH_LOG(
    MemcpyHostToDeviceAsync(embedding_cache_table_manager.embedding_device_cache_->hash_swap_index_addr_,
                            device_cache_host_to_device_ids, swap_indices_size * sizeof(int), device_context_,
                            stream_id_),
    "Memcpy host to device asynchronously failed.");

  RETURN_IF_FALSE_WITH_LOG(
    UpdateDeviceCache(embedding_cache_table_manager.embedding_device_cache_->hash_swap_index_addr_,
                      embedding_cache_table_manager.embedding_device_cache_->hash_swap_value_addr_, swap_indices_size,
                      cache_vocab_size, embedding_size, hash_table_addr, hash_info.device_address),
    "Update device embedding cache failed.");
  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->device_res_manager_);
  RETURN_IF_FALSE_WITH_LOG(device_context_->device_res_manager_->SyncStream(stream_id_), "Synchronize stream failed.");
  return true;
}

void DeviceSparseEmbeddingOperation::GetRemoteEmbeddingSliceBound(
  size_t vocab_size, size_t server_num, std::vector<std::pair<size_t, size_t>> *remote_embedding_slice_bounds) {
  if (server_num != 1) {
    MS_LOG(EXCEPTION)
      << "Sparse mode does not support multiple servers currently, so server number should be 1, but got: "
      << server_num;
  }

  MS_EXCEPTION_IF_NULL(remote_embedding_slice_bounds);
  // Sparse mode does not support multiple servers currently, so the id does not need to be split, and the id range is
  // specified from 0 to INTMAX .
  remote_embedding_slice_bounds->emplace_back(0, INT32_MAX);
}

void DeviceSparseEmbeddingOperation::BuildEmbeddingCacheLookupKernel() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id((std::numeric_limits<uint32_t>::max)());
  embedding_cache_graphs_.push_back(graph);

  // 1. Create parameter nodes which are inputs of embedding cache look up kernel(operator name: 'MapTensorGet').
  ParameterPtr input_param = NewMapParameter(graph, kNumberTypeInt32, kNumberTypeFloat32, kOneDimensionalShape);
  ParameterPtr input_ids = NewParameter(graph, kInt32, kOneDimensionalShape);

  // 2. Create a CNode for operator MapTensorGet.
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kMapTensorGetOpName);
  emb_lookup_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrOutputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrInsertDefaultValue, MakeValue(false));

  std::vector<AnfNodePtr> emb_lookup_input_nodes{mindspore::NewValueNode(emb_lookup_primitive), input_param, input_ids};
  embedding_cache_lookup_node_ = graph->NewCNode(emb_lookup_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_lookup_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->kernel_executor_);
  device_context_->kernel_executor_->CreateKernel({embedding_cache_lookup_node_});
  AnfAlgo::SetStreamId(stream_id_, embedding_cache_lookup_node_.get());
}

void DeviceSparseEmbeddingOperation::BuildEmbeddingCacheUpdateKernel() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id((std::numeric_limits<uint32_t>::max)());
  embedding_cache_graphs_.push_back(graph);

  // 1. Create parameter nodes which are inputs of embedding cache update kernel(operator name: 'MapTensorPut').
  ParameterPtr input_param = NewMapParameter(graph, kNumberTypeInt32, kNumberTypeFloat32, kOneDimensionalShape);
  ParameterPtr input_ids = NewParameter(graph, kInt32, kOneDimensionalShape);
  ParameterPtr update_values = NewParameter(graph, kFloat32, kTwoDimensionalShape);

  // 2. Create a CNode for operator MapTensorPut.
  PrimitivePtr embedding_cache_update_primitive = std::make_shared<Primitive>(kMapTensorPutOpName);
  embedding_cache_update_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));

  std::vector<AnfNodePtr> embedding_cache_update_input_nodes{mindspore::NewValueNode(embedding_cache_update_primitive),
                                                             input_param, input_ids, update_values};
  embedding_cache_update_node_ = graph->NewCNode(embedding_cache_update_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node_);
  embedding_cache_update_node_->set_abstract(input_param->abstract());

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->kernel_executor_);
  device_context_->kernel_executor_->CreateKernel({embedding_cache_update_node_});
  AnfAlgo::SetStreamId(stream_id_, embedding_cache_update_node_.get());
}

void DeviceSparseEmbeddingOperation::BuildEmbeddingCacheEraseKernel() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id((std::numeric_limits<uint32_t>::max)());
  embedding_cache_graphs_.push_back(graph);

  // 1. Create parameter nodes which are inputs of embedding cache erase kernel(operator name: 'MapTensorErase').
  ParameterPtr input_param = NewMapParameter(graph, kNumberTypeInt32, kNumberTypeFloat32, kOneDimensionalShape);
  ParameterPtr input_ids = NewParameter(graph, kInt32, kOneDimensionalShape);

  // 2. Create a CNode for operator MapTensorErase.
  PrimitivePtr embedding_cache_erase_primitive = std::make_shared<Primitive>(kMapTensorEraseOpName);
  embedding_cache_erase_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  embedding_cache_erase_primitive->set_attr(kAttrOutputIsDynamicShape, MakeValue(true));

  std::vector<AnfNodePtr> embedding_cache_erase_input_nodes{mindspore::NewValueNode(embedding_cache_erase_primitive),
                                                            input_param, input_ids};
  embedding_cache_erase_node_ = graph->NewCNode(embedding_cache_erase_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_erase_node_);
  embedding_cache_erase_node_->set_abstract(input_param->abstract());

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->kernel_executor_);
  device_context_->kernel_executor_->CreateKernel({embedding_cache_erase_node_});
  AnfAlgo::SetStreamId(stream_id_, embedding_cache_erase_node_.get());
}

ParameterPtr DeviceSparseEmbeddingOperation::NewMapParameter(const KernelGraphPtr &graph, TypeId key_type,
                                                             TypeId value_type, const ShapeVector &value_shape) {
  MS_EXCEPTION_IF_NULL(graph);

  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  auto map_tensor = std::make_shared<tensor::MapTensor>(key_type, value_type, value_shape, nullptr);
  auto abstract = std::make_shared<abstract::AbstractMapTensor>(map_tensor);
  param->set_abstract(abstract);

  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  std::vector<std::string> formats = {kOpFormat_DEFAULT};
  std::vector<TypeId> types = {kObjectTypeMapTensorType};
  kernel_build_info_builder->SetOutputsFormat(formats);
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());

  auto mutable_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(mutable_inputs);
  mutable_inputs->push_back(param);

  return param;
}

bool DeviceSparseEmbeddingOperation::LookupDeviceCache(const DeviceAddress *embed_device_address, void *ids,
                                                       void *embedding_cache, size_t ids_num, size_t cache_size,
                                                       size_t embedding_size, void *outputs) {
  MS_ERROR_IF_NULL(embed_device_address);
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(embedding_cache_lookup_node_);

  AddressPtrList kernel_inputs = {std::make_shared<Address>(embedding_cache, 1),
                                  std::make_shared<Address>(ids, ids_num * sizeof(int))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(outputs, ids_num * embedding_size * sizeof(float))};

  // The user data contain a device hash table which is the device cache to update.
  UserDataPtr user_data = embed_device_address->user_data();
  MS_ERROR_IF_NULL(user_data);
  auto kernel_mod = AnfAlgo::GetKernelMod(embedding_cache_lookup_node_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_input_user_data(user_data.get(), 0);

  // Do embedding cache look up on device.
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

bool DeviceSparseEmbeddingOperation::UpdateDeviceCache(void *ids, void *update_value, size_t ids_num, size_t cache_size,
                                                       size_t embedding_size, void *embedding_cache,
                                                       DeviceAddress *embed_device_address) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(update_value);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(embed_device_address);
  MS_ERROR_IF_NULL(embedding_cache_update_node_);

  AddressPtrList kernel_inputs = {std::make_shared<Address>(embedding_cache, 1),
                                  std::make_shared<Address>(ids, ids_num * sizeof(int)),
                                  std::make_shared<Address>(update_value, ids_num * embedding_size * sizeof(float))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(embedding_cache, 1)};

  // The user data contain a device hash table which is the device cache to update.
  UserDataPtr user_data = embed_device_address->user_data();
  MS_ERROR_IF_NULL(user_data);
  auto kernel_mod = AnfAlgo::GetKernelMod(embedding_cache_update_node_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_input_user_data(user_data.get(), 0);

  // Do update cache on device.
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

bool DeviceSparseEmbeddingOperation::EraseDeviceCache(void *ids, size_t ids_num, void *embedding_cache,
                                                      DeviceAddress *embed_device_address) {
  MS_ERROR_IF_NULL(embed_device_address);
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(embedding_cache_erase_node_);

  AddressPtrList kernel_inputs = {std::make_shared<Address>(embedding_cache, 1),
                                  std::make_shared<Address>(ids, ids_num * sizeof(int))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(embedding_cache, 1)};

  // The user data contain a device hash table which is the device cache to update.
  UserDataPtr user_data = embed_device_address->user_data();
  MS_ERROR_IF_NULL(user_data);
  auto kernel_mod = AnfAlgo::GetKernelMod(embedding_cache_erase_node_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_input_user_data(user_data.get(), 0);

  // Erase embedding cache on device.
  MS_ERROR_IF_NULL(device_context_);
  MS_ERROR_IF_NULL(device_context_->kernel_executor_);
  auto ret = device_context_->kernel_executor_->LaunchKernel(embedding_cache_erase_node_, kernel_inputs, {},
                                                             kernel_outputs, stream_id_);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_erase_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

bool DeviceSparseEmbeddingOperation::CheckCacheHit(const int *batch_ids, const size_t batch_ids_num, bool *in_device,
                                                   size_t data_step) const {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(in_device);

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
    threads[i] = std::thread(&DeviceSparseEmbeddingOperation::CheckCacheHitFunc, this, batch_ids + offset, proc_len,
                             in_device + offset, hash_hit_count + i, data_step);
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

bool DeviceSparseEmbeddingOperation::CheckCacheHitFunc(const int *batch_ids, const size_t batch_ids_num,
                                                       bool *in_device, size_t *hash_hit_count,
                                                       size_t data_step) const {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(hash_hit_count);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  auto &device_hash_map = embedding_cache_table_manager.embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();

  // Count how many feature ids reside in the device cache.
  for (size_t i = 0; i < batch_ids_num; ++i) {
    auto iter = hash_id_to_index.find(batch_ids[i]);
    if (iter != hash_id_to_index.end()) {
      if (device_hash_map->hash_step(iter->second) != data_step) {
        ++(*hash_hit_count);
        device_hash_map->set_hash_step(iter->second, data_step);
      }
      in_device[i] = true;
    }
  }
  return true;
}

bool DeviceSparseEmbeddingOperation::ParseDeviceData(int id, bool *need_swap_device_to_host,
                                                     bool *need_swap_host_to_device, size_t data_step,
                                                     size_t graph_running_step, bool *device_cache_need_wait_graph) {
  MS_ERROR_IF_NULL(need_swap_device_to_host);
  MS_ERROR_IF_NULL(need_swap_host_to_device);
  MS_ERROR_IF_NULL(embedding_cache_table_manager.embedding_device_cache_);
  auto &device_hash_map = embedding_cache_table_manager.embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);

  int index;
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
    int *device_to_host_index = embedding_cache_table_manager.embedding_device_cache_->device_to_host_index.get();
    int *device_to_host_ids = embedding_cache_table_manager.embedding_device_cache_->device_to_host_ids.get();
    int *host_to_device_ids = embedding_cache_table_manager.embedding_device_cache_->host_to_device_ids.get();
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
      host_to_device_ids[statistics_info_->host_to_device_size_] = id;
      statistics_info_->host_to_device_size_++;
      *need_swap_device_to_host = statistics_info_->device_to_host_size_ > tmp_device_to_host_size;
      break;
    }
  }

  return true;
}
}  // namespace runtime
}  // namespace mindspore
