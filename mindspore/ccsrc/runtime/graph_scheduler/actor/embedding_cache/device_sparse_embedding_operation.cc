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
#include <limits>
#include "runtime/graph_scheduler/actor/embedding_cache/device_sparse_embedding_operation.h"

namespace mindspore {
namespace runtime {
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
    RETURN_IF_FALSE_WITH_LOG(
      ParseDeviceData(batch_ids[i], &need_swap_device_to_host, &need_swap_host_to_device, data_step),
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

  // 3. Replace the batch_ids by hash index for GetNext operator to get hash index as input.
  size_t data_size = batch_ids_num * sizeof(int);
  size_t dest_len = data_size;
  void *data = reinterpret_cast<void *>(batch_ids);
  ret = memcpy_s(data, dest_len, batch_ids, data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy hash index failed, errno[" << ret << "]";
    return false;
  }
  return true;
}

void DeviceSparseEmbeddingOperation::BuildEmbeddingCacheLookupKernel() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id((std::numeric_limits<uint32_t>::max)());
  embedding_cache_graphs_.push_back(graph);

  // 1. Create parameter nodes which are inputs of embedding cache look up kernel(operator name: 'MapTensorGet').
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_ids = NewParameter(graph, kInt32, kOneDimensionalShape);

  // 2. Create a CNode for operator MapTensorGet.
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kMapTensorGetOpName);
  emb_lookup_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrOutputIsDynamicShape, MakeValue(true));

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
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_ids = NewParameter(graph, kInt32, kOneDimensionalShape);
  ParameterPtr update_values = NewParameter(graph, kFloat32, kTwoDimensionalShape);

  // 2. Create a CNode for operator MapTensorPut.
  PrimitivePtr embedding_cache_update_primitive = std::make_shared<Primitive>(kMapTensorPutOpName);
  embedding_cache_update_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));

  std::vector<AnfNodePtr> embedding_cache_update_input_nodes{mindspore::NewValueNode(embedding_cache_update_primitive),
                                                             input_param, input_ids, update_values};
  embedding_cache_update_node_ = graph->NewCNode(embedding_cache_update_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_update_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_EXCEPTION_IF_NULL(device_context_->kernel_executor_);
  device_context_->kernel_executor_->CreateKernel({embedding_cache_update_node_});
  AnfAlgo::SetStreamId(stream_id_, embedding_cache_update_node_.get());
}

bool DeviceSparseEmbeddingOperation::CheckCacheHit(const int *batch_ids, const size_t batch_ids_num, bool *in_device,
                                                   size_t data_step) {
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
                                                       bool *in_device, size_t *hash_hit_count, size_t data_step) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(hash_hit_count);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
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
                                                     bool *need_swap_host_to_device, size_t data_step) {
  MS_ERROR_IF_NULL(need_swap_device_to_host);
  MS_ERROR_IF_NULL(need_swap_host_to_device);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);

  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    *need_swap_device_to_host = false;
    *need_swap_host_to_device = false;
    if (device_hash_map->hash_step(id) != data_step) {
      statistics_info_->hash_hit_count_++;
      device_hash_map->set_hash_step(id, data_step);
    }
  } else {
    int *host_to_device_index = embedding_device_cache_->host_to_device_index.get();
    int *host_to_device_ids = embedding_device_cache_->host_to_device_ids.get();
    MS_ERROR_IF_NULL(host_to_device_index);
    MS_ERROR_IF_NULL(host_to_device_ids);
    auto tmp_device_to_host_size = statistics_info_->device_to_host_size_;

    host_to_device_index[statistics_info_->host_to_device_size_] = id;
    host_to_device_ids[statistics_info_->host_to_device_size_] = id;
    statistics_info_->host_to_device_size_++;
    *need_swap_device_to_host = statistics_info_->device_to_host_size_ > tmp_device_to_host_size;
  }
  return true;
}

bool DeviceSparseEmbeddingOperation::LookupDeviceCache(void *ids, void *embedding_cache, size_t ids_num,
                                                       size_t cache_size, size_t embedding_size, void *outputs) {
  MS_ERROR_IF_NULL(ids);
  MS_ERROR_IF_NULL(embedding_cache);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(embedding_cache_lookup_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, kInputIndexZero);
  MS_ERROR_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_ids_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, kInputIndexOne);
  MS_ERROR_IF_NULL(input_ids_node);
  const ShapeVector input_indices_shape = {SizeToLong(ids_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_ids_node->set_abstract(input_indices_abstract);

  // 2. Infer shape for embedding cache look up kernel(operator name: 'MapTensorGet') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_lookup_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_lookup_node_->fullname_with_scope();
    return false;
  }

  // 3. Do embedding cache look up on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(ids, ids_num * sizeof(int))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(outputs, ids_num * embedding_size * sizeof(float))};

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
}  // namespace runtime
}  // namespace mindspore
