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

#include "ps/worker.h"
#include "pipeline/jit/pipeline.h"

namespace mindspore {
namespace ps {
void Worker::Run() {
  std::lock_guard<std::mutex> lock(running_mutex_);

  server_num_ = PSContext::instance()->initial_server_num();
  if (running_) {
    MS_LOG(INFO) << "'Worker is already running.";
    return;
  }
  if (!PSContext::instance()->is_worker()) {
    MS_LOG(EXCEPTION) << "The role is not worker.";
  }

  Initialize();
  worker_node_.set_event_callback([&](const core::NodeEvent &event) {
    if ((event == core::NodeEvent::CLUSTER_TIMEOUT) ||
        (event == core::NodeEvent::SCHEDULER_TIMEOUT || (event == core::NodeEvent::NODE_TIMEOUT))) {
      MS_LOG(WARNING) << "Trigger timeout event:" << event << " begin to exit the system!";
      Finalize();
      exit(0);
    }
  });
  MS_LOG(INFO) << "Worker starts connecting to scheduler and server...";
  worker_node_.Start();
  MS_LOG(INFO) << "Worker connected successfully.";

  running_ = true;
}

void Worker::Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const ShapeVector &sizes) {
  if (keys.size() == 0) {
    MS_LOG(EXCEPTION) << "key size should be greater than zero";
  }
  if (key_to_optimId_.count(keys[0]) == 0) {
    MS_LOG(EXCEPTION) << "no optim id found for key" << keys[0];
  }
  Key key = keys[0];
  int64_t optim_id = key_to_optimId_[key];
  MS_LOG(INFO) << "The key is:" << key << " the optim_id:" << optim_id;
  bool is_sparse = false;
  if (optim_id == 1 || optim_id == 2 || optim_id == 3) {
    is_sparse = true;
  }
  int64_t grad_index = -1;
  int64_t indice_index = -1;

  // Sparse adam gradient
  if (optim_id == 1 || optim_id == 2) {
    grad_index = 6;
    indice_index = 7;

    // Sparse ftrl gradient
  } else if (optim_id == 3) {
    grad_index = 0;
    indice_index = 1;
  }

  size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0, std::plus<int64_t>());
  std::vector<float> total_buffer(total_size, 0);
  size_t offset = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    void *dst_data = total_buffer.data() + offset / sizeof(float);
    void *src_data = reinterpret_cast<void *>(addrs[i]);
    MS_EXCEPTION_IF_NULL(dst_data);
    MS_EXCEPTION_IF_NULL(src_data);
    int size = sizes[i] * sizeof(float);
    size_t dest_size = IntToSize(size);
    size_t src_size = IntToSize(size);
    auto ret = memcpy_s(dst_data, dest_size, src_data, src_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
    offset += size;
  }
  MS_LOG(INFO) << "The total size is:" << total_size;

  while (running_ && (!IsReadyForPush(keys[0]))) {
    continue;
  }
  std::vector<int> sizes_int;
  (void)std::transform(sizes.begin(), sizes.end(), std::back_inserter(sizes_int),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (!is_sparse) {
    PushData(std::vector<Key>(keys), total_buffer, std::vector<int>(sizes_int), kPushCmd);
  } else {
    std::vector<int64_t> &var_shape = key_to_optim_shapes_[key][0];
    int64_t first_dim_size = var_shape[0];
    int64_t outer_dim_size = std::accumulate(var_shape.begin() + 1, var_shape.end(), 1, std::multiplies<int64_t>());
    MS_LOG(DEBUG) << "The keys:" << keys << " the total_buffer:" << total_buffer << " the sizes_int:" << sizes_int
                  << " the grad_index:" << grad_index << " the indice_index:" << indice_index
                  << " the first_dim_size:" << first_dim_size << " the outer_dim_size" << outer_dim_size;
    PushSparseData(std::vector<Key>(keys), total_buffer, std::vector<int>(sizes_int), grad_index, indice_index,
                   first_dim_size, outer_dim_size);
  }
}

void Worker::Pull(const size_t key, void *dev_addr, const size_t size) {
  MS_EXCEPTION_IF_NULL(dev_addr);
  std::vector<float> variables(size / sizeof(float), 0);
  while (running_ && (!IsReadyForPull(key))) {
    continue;
  }
  PullData({key}, &variables, nullptr, kPullCmd);
  MS_LOG(DEBUG) << "The variables:" << variables << " the size is:" << size;
  size_t dst_size = size;
  size_t src_size = size;
  auto ret = memcpy_s(dev_addr, dst_size, variables.data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
}

size_t Worker::SetParamKey(const std::string &param_name) {
  size_t key = UINT64_MAX;
  if (param_to_key_.count(param_name)) {
    key = param_to_key_[param_name];
    MS_LOG(INFO) << param_name << " key is already set: key value is " << key;
  } else {
    key = key_cnt_++;
    param_to_key_[param_name] = key;
    MS_LOG(INFO) << "Set key " << key << " for parameter " << param_name;
  }
  return key;
}

size_t Worker::GetParamKey(const std::string &param_name) {
  size_t key = kInvalidKey;
  if (param_to_key_.find(param_name) != param_to_key_.end()) {
    key = param_to_key_[param_name];
    MS_LOG(DEBUG) << "Get key of parameter " << param_name << " key is " << key;
  }
  return key;
}

void Worker::SetParamInitInServer(const std::string &param_name, bool init_in_server) {
  MS_LOG(INFO) << "Set parameter " << param_name << " init_in_server:" << init_in_server;
  param_to_init_in_server_[param_name] = init_in_server;
}

bool Worker::GetParamInitInServer(const std::string &param_name) {
  if (param_to_init_in_server_.count(param_name) == 0) {
    return false;
  }
  return param_to_init_in_server_[param_name];
}

void Worker::SetKeyOptimId(size_t key, const std::string &optimizer_name) {
  MS_LOG(INFO) << "SetKeyOptimId key is:" << key << " optimizer_name:" << optimizer_name;
  key_to_optimId_[key] = Util::optimizer_id(optimizer_name);
}

void Worker::SetOptimInputShapes(size_t key, const ShapeVector &shape) {
  if (key_to_optim_shapes_.find(key) == key_to_optim_shapes_.end()) {
    key_to_optim_shapes_[key] = {shape};
  } else {
    key_to_optim_shapes_[key].push_back(shape);
  }
}

void Worker::AddEmbeddingTable(const Key &key, const size_t &row_count) {
  bool has_init = IsKeyInit(key);
  if (has_init) {
    return;
  }
  uint64_t begin = 0;
  uint64_t end = 0;
  for (int64_t i = 0; i < server_num_; i++) {
    int64_t local_row_cnt = Util::LocalShard(row_count, i, server_num_);
    MS_LOG(DEBUG) << "The row_count:" << row_count << " the local_row_cnt:" << local_row_cnt;
    if (i == 0) {
      end = local_row_cnt - 1;
    } else {
      begin = end + 1;
      end += local_row_cnt;
    }
    EmbeddingTableShardMetadata range(begin, end);
    if (embedding_table_ranges_.count(key) == 0) {
      embedding_table_ranges_[key] = std::make_shared<std::vector<EmbeddingTableShardMetadata>>();
      MS_EXCEPTION_IF_NULL(embedding_table_ranges_[key]);
    }
    embedding_table_ranges_[key]->push_back(range);
  }
  embedding_row_cnt_[key] = row_count;
}

void Worker::InitPSEmbeddingTable(const size_t &key, const std::vector<size_t> &input_shape,
                                  const std::vector<size_t> &indices_shape, const std::vector<size_t> &output_shape,
                                  const ParamInitInfoMessage &info) {
  bool has_init = IsKeyInit(key);
  if (has_init) {
    MS_LOG(DEBUG) << "The key embedding table of key " << key << " is initialized.";
    return;
  }

  EmbeddingTableMeta embedding_table_meta;
  embedding_table_meta.set_key(key);
  *embedding_table_meta.mutable_input_shape() = {input_shape.begin(), input_shape.end()};
  *embedding_table_meta.mutable_indices_shape() = {indices_shape.begin(), indices_shape.end()};
  *embedding_table_meta.mutable_output_shape() = {output_shape.begin(), output_shape.end()};
  *embedding_table_meta.mutable_info() = info;

  std::string kv_data = embedding_table_meta.SerializeAsString();

  std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
  size_t dest_size = kv_data.length();
  int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }

  worker_node_.Broadcast(core::NodeRole::SERVER, res, kv_data.length(), kInitEmbeddingsCmd);
}

void Worker::InitPSParamAndOptim(const AnfNodePtr &input_node, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(input_node);
  auto pk_node = input_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(pk_node);
  const std::string &param_name = pk_node->fullname_with_scope();
  void *param_data = tensor->data_c();
  size_t param_size = LongToSize(tensor->data().nbytes());

  size_t param_key = GetParamKey(param_name);
  if (param_key == kInvalidKey) {
    MS_LOG(DEBUG) << "Parameter " << param_name << " has no key assigned.";
    return;
  }
  bool init_in_server = false;
  auto param_info_ptr = pk_node->param_info();
  if (param_info_ptr != nullptr && param_info_ptr->init_in_server()) {
    init_in_server = true;
  }
  SetParamInitInServer(param_name, init_in_server);
  bool init = IsKeyInit(param_key);
  if (!init) {
    MS_LOG(INFO) << "Init parameter key " << param_key << " and optimizer in parameter server side for " << param_name
                 << ", whether init in server: " << init_in_server;
    AddKeyToServerId(param_key);
    if (!PsDataPrefetch::GetInstance().cache_enable()) {
      if (!init_in_server) {
        if (param_size > INT_MAX) {
          MS_LOG(EXCEPTION) << "PS mode max weight size is " << INT_MAX << ", " << param_name << " size is "
                            << param_size;
        }
        InitPSParamData({param_key}, param_data, param_size);
      }
      InitPSOptimId(param_key);
      InitPSOptimInputShapes(param_key);
    }
  }
}

void Worker::DoPSEmbeddingLookup(const Key &key, const std::vector<int> &lookup_ids, std::vector<float> *lookup_result,
                                 int64_t cmd) {
  MS_EXCEPTION_IF_NULL(lookup_result);
  EmbeddingTableLookup embedding_table_lookup;
  embedding_table_lookup.set_key(key);
  *embedding_table_lookup.mutable_keys() = {lookup_ids.begin(), lookup_ids.end()};

  PartitionEmbeddingMessages messages;
  lookup_partitioner_(embedding_table_lookup, &messages, {});
  std::vector<uint32_t> rank_ids;
  std::vector<DataPtr> data;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < messages.size(); i++) {
    if (messages.at(i).first) {
      rank_ids.push_back(i);
      std::string kv_data = messages.at(i).second.SerializeAsString();

      std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
      size_t dest_size = kv_data.length();
      int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      data.push_back(res);
      sizes.push_back(kv_data.length());
    }
  }

  std::vector<VectorPtr> resp;
  worker_node_.Send(core::NodeRole::SERVER, rank_ids, data, sizes, cmd, &resp);
  int64_t single_id_len = SizeToLong(lookup_result->size() / lookup_ids.size());
  std::unordered_map<Key, std::shared_ptr<std::pair<float *, int64_t>>> id_addr_map;
  std::shared_ptr<std::vector<float>> values = std::make_shared<std::vector<float>>();
  std::shared_ptr<std::vector<Key>> keys = std::make_shared<std::vector<Key>>();
  int64_t value_offset = 0;
  for (size_t i = 0; i < resp.size(); ++i) {
    KVMessage message;
    message.ParseFromArray(resp.at(i)->data(), resp.at(i)->size());
    for (auto j = 0; j < message.values_size(); j++) {
      values->push_back(message.values(j));
    }
    for (auto k = 0; k < message.keys_size(); k++) {
      const Key &key = message.keys(k);
      keys->push_back(key);
    }
  }

  for (size_t i = 0; i < keys->size(); i++) {
    const Key &key = keys->at(i);
    float *addr = values->data() + value_offset;
    value_offset += single_id_len;
    id_addr_map[key] = std::make_shared<std::pair<float *, int64_t>>(std::make_pair(addr, single_id_len));
  }

  float *result_addr = lookup_result->data();
  MS_EXCEPTION_IF_NULL(result_addr);
  int64_t offset = 0;
  size_t dst_size = 0;
  size_t src_size = 0;
  void *dst_data = nullptr;
  void *src_data = nullptr;
  for (size_t i = 0; i < lookup_ids.size(); i++) {
    if (id_addr_map.count(lookup_ids[i]) == 0) {
      offset += single_id_len;
      continue;
    }
    const Key &key = static_cast<Key>(lookup_ids[i]);
    auto &pair = id_addr_map[key];
    int64_t size = single_id_len * sizeof(float);
    dst_size = size;
    src_size = size;
    dst_data = result_addr + offset;
    src_data = pair->first;
    MS_EXCEPTION_IF_NULL(dst_data);
    MS_EXCEPTION_IF_NULL(src_data);
    auto ret = memcpy_s(dst_data, dst_size, src_data, src_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
    offset += single_id_len;
  }
}

void Worker::UpdateEmbeddingTable(const std::vector<Key> &keys, const std::vector<int> &lookup_ids,
                                  const std::vector<float> &vals) {
  KVMessage kvs;
  *kvs.mutable_keys() = {keys.begin(), keys.end()};
  *kvs.mutable_len() = {lookup_ids.begin(), lookup_ids.end()};
  *kvs.mutable_values() = {vals.begin(), vals.end()};
  PartitionKVMessages messages;
  update_embedding_partitioner_(kvs, &messages, {});
  std::vector<uint32_t> rank_ids;
  std::vector<DataPtr> data;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < messages.size(); i++) {
    if (messages.at(i).first) {
      rank_ids.push_back(i);
      std::string kv_data = messages.at(i).second.SerializeAsString();

      std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
      size_t dest_size = kv_data.length();
      int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      data.push_back(res);
      sizes.push_back(kv_data.length());
    }
  }
  worker_node_.Send(core::NodeRole::SERVER, rank_ids, data, sizes, kUpdateEmbeddingsCmd);
}

void Worker::Finalize() {
  if (running_) {
    MS_LOG(INFO) << "Worker starts finalizing...";
    KVMessage kvs;
    kvs.add_keys(0);
    kvs.add_values(0.0f);
    std::string kv_data = kvs.SerializeAsString();
    std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
    size_t dest_size = kv_data.length();
    int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
    worker_node_.Broadcast(core::NodeRole::SERVER, res, kv_data.length(), kFinalizeCmd);
    worker_node_.Finish();
    worker_node_.Stop();
    running_ = false;
    MS_LOG(INFO) << "Worker finalized successfully.";
  }
}

void Worker::Initialize() {
  lookup_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    LookupIdPartitioner(send, partition, attrs);
  };
  worker_init_embedding_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    WorkerInitEmbeddingPartitioner(send, partition, attrs);
  };
  round_robin_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    RoundRobinPartitioner(send, partition, attrs);
  };
  sparse_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    SparsePartitioner(send, partition, attrs);
  };
  update_embedding_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    UpdateEmbeddingPartitioner(send, partition, attrs);
  };
  broadcast_partitioner_ = [this](auto &&send, auto &&partition, auto &&attrs) {
    BroadcastPartitioner(send, partition, attrs);
  };
}

bool Worker::IsKeyInit(const size_t key) {
  if (init_keys_.find(key) == init_keys_.end() || !init_keys_[key]) {
    return false;
  }
  return true;
}

void Worker::AddKeyToServerId(const Key &key) { AddKeyByHashMod(key); }

void Worker::AddKeyByHashMod(const Key &key) {
  if (server_num_ == 0) {
    MS_LOG(EXCEPTION) << "Server number is invalid:0";
  }
  key_to_server_id_[key] = static_cast<int64_t>(key % server_num_);
  MS_LOG(INFO) << "The server id of key " << key << " is " << key_to_server_id_[key];
}

void Worker::InitPSOptimId(const size_t param_key) {
  MS_LOG(INFO) << "InitPSOptimId key is:" << param_key;
  if (key_to_optimId_.count(param_key) == 0) {
    MS_LOG(EXCEPTION) << "Can't find optimizer id of parameter key " << param_key;
  }
  int64_t optim_id = key_to_optimId_[param_key];

  std::vector<Key> keys = {param_key};
  std::vector<float> optim_id_vals = {static_cast<float>(optim_id)};
  std::vector<int> optim_id_lens = {SizeToInt(optim_id_vals.size())};
  MS_LOG(INFO) << "The keys is" << keys << " the optim_id_vals is: " << optim_id_vals
               << " optim_id_lens is:" << optim_id_lens;
  PushData(keys, optim_id_vals, optim_id_lens, kInitWeightToOptimIdCmd);
}

void Worker::InitPSOptimInputShapes(const size_t key) {
  std::vector<Key> keys;
  std::vector<int> shape_len;
  std::vector<float> all_shape;
  std::vector<ShapeVector> shapes = key_to_optim_shapes_[key];
  for (auto shape : shapes) {
    keys.push_back(key);
    if (shape.size() == 0) {
      shape_len.push_back(1);
      all_shape.push_back(1);
    } else {
      shape_len.push_back(SizeToLong(shape.size()));
      std::transform(shape.begin(), shape.end(), std::back_inserter(all_shape),
                     [](size_t dim) -> float { return static_cast<float>(dim); });
    }
  }
  MS_LOG(INFO) << "keys:" << keys;
  MS_LOG(INFO) << "shape_len:" << shape_len;
  MS_LOG(INFO) << "all_shape:" << all_shape;
  if (!init_keys_[key]) {
    init_keys_[key] = true;
  }
  PushData(keys, all_shape, shape_len, kInitOptimInputsShapeCmd);
}

void Worker::InitPSParamData(const std::vector<size_t> &keys, void *const origin_addr, size_t size) {
  MS_EXCEPTION_IF_NULL(origin_addr);
  std::vector<float> addr{reinterpret_cast<float *>(origin_addr),
                          reinterpret_cast<float *>(origin_addr) + size / sizeof(float)};
  std::vector<Key> key(keys);
  std::vector<int> lens;
  lens.push_back(addr.size());
  MS_LOG(INFO) << "the keys are:" << keys;
  MS_LOG(INFO) << "the values are:" << addr;
  PushData(key, addr, lens, kInitWeightsCmd);
  init_keys_[key[0]] = true;
}

bool Worker::IsReadyForPush(const Key &key) {
  std::vector<float> result(1, 0);
  PullData({key}, &result, nullptr, kCheckReadyForPushCmd);
  MS_LOG(INFO) << "key:" << key;
  if (result[0] > 0) {
    MS_LOG(INFO) << "IsReadyForPush:";
    return true;
  } else {
    MS_LOG(INFO) << "IsReadyForPush:";
    return false;
  }
}

bool Worker::IsReadyForPull(const Key &key) {
  std::vector<float> result(1, 0);
  PullData({key}, &result, nullptr, kCheckReadyForPullCmd);
  if (result[0] > 0) {
    MS_LOG(INFO) << "IsReadyForPull";
    return true;
  } else {
    MS_LOG(INFO) << "IsReadyForPull";
    return false;
  }
}

void Worker::PrepareSparseGradient(const size_t begin, const size_t end, const std::unordered_set<int> &distinct_ids,
                                   const std::vector<std::pair<int, float *>> &indice_to_grads, const int *all_indice,
                                   const size_t segment_size, float *gradient, int *indices) {
  MS_EXCEPTION_IF_NULL(all_indice);
  MS_EXCEPTION_IF_NULL(gradient);
  MS_EXCEPTION_IF_NULL(indices);
  int64_t offset = 0;
  int64_t index = 0;
  size_t segment_data_size = segment_size * sizeof(float);
  size_t dst_size;
  size_t src_size;
  void *dst_data = nullptr;
  void *src_data = nullptr;
  for (auto &pair : indice_to_grads) {
    if (distinct_ids.count(pair.first) == 0) {
      continue;
    }
    indices[index++] = pair.first;

    dst_size = segment_data_size;
    src_size = segment_data_size;
    dst_data = gradient + offset;
    src_data = pair.second;
    MS_EXCEPTION_IF_NULL(dst_data);
    MS_EXCEPTION_IF_NULL(src_data);
    auto ret = memcpy_s(gradient + offset, dst_size, pair.second, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
    offset += segment_size;
  }
}

void Worker::BuildSparseValue(const std::vector<int> &lengths, const size_t grad_index, const size_t indice_index,
                              const float *original_data, const float *grads, int *indices,
                              std::vector<float> *reduced_data) {
  MS_EXCEPTION_IF_NULL(original_data);
  MS_EXCEPTION_IF_NULL(grads);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(reduced_data);
  int64_t offset = 0;
  size_t dst_size = 0;
  size_t src_size = 0;
  void *dst_data = nullptr;
  void *src_data = nullptr;
  for (size_t i = 0; i < lengths.size(); i++) {
    if (i != grad_index && i != indice_index) {
      int data_size = lengths[i] * sizeof(float);
      dst_size = data_size;
      src_size = data_size;
      dst_data = reduced_data->data() + offset;
      src_data = const_cast<float *>(original_data) + offset;
      MS_EXCEPTION_IF_NULL(dst_data);
      MS_EXCEPTION_IF_NULL(src_data);
      auto ret = memcpy_s(dst_data, dst_size, src_data, src_size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
    }
    offset += lengths[i];
  }

  // Fill the reduced gradient
  int64_t grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  int64_t data_size = lengths[grad_index] * sizeof(float);
  dst_size = data_size;
  src_size = data_size;
  dst_data = reduced_data->data() + grad_offset;
  src_data = const_cast<float *>(grads);
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  auto ret = memcpy_s(dst_data, dst_size, src_data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }

  // Fill the reduced indice
  int64_t indice_offset = grad_offset + lengths[grad_index];
  data_size = lengths[indice_index] * sizeof(float);
  float *indice_data = reduced_data->data() + indice_offset;
  dst_size = data_size;
  src_size = data_size;
  dst_data = indice_data;
  src_data = indices;
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  ret = memcpy_s(dst_data, dst_size, src_data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
}

void Worker::PushData(const std::vector<Key> &keys, const std::vector<float> &vals, const std::vector<int> &lens,
                      int cmd, int64_t priority) {
  KVMessage kvs;
  *kvs.mutable_keys() = {keys.begin(), keys.end()};
  *kvs.mutable_values() = {vals.begin(), vals.end()};
  *kvs.mutable_len() = {lens.begin(), lens.end()};
  MS_LOG(INFO) << "the result is:" << embedding_table_ranges_.count(keys[0]);
  if (embedding_table_ranges_.count(keys[0])) {
    if (cmd == kInitWeightsCmd) {
      SendForPush(cmd, kvs, worker_init_embedding_partitioner_, {});
    } else {
      std::string kv_data = kvs.SerializeAsString();
      std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
      size_t dest_size = kv_data.length();
      int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      worker_node_.Broadcast(core::NodeRole::SERVER, res, kv_data.length(), cmd);
    }
  } else {
    SendForPush(cmd, kvs, round_robin_partitioner_, {});
  }
}

void Worker::PushSparseData(const std::vector<Key> &keys, const std::vector<float> &vals, const std::vector<int> &lens,
                            size_t grad_index, size_t indice_index, size_t first_dim_size, size_t outer_dim_size) {
  KVMessage kvs;
  *kvs.mutable_keys() = {keys.begin(), keys.end()};
  *kvs.mutable_values() = {vals.begin(), vals.end()};
  *kvs.mutable_len() = {lens.begin(), lens.end()};
  if (embedding_table_ranges_.count(keys[0])) {
    std::map<int64_t, int64_t> attrs{{0, grad_index}, {1, indice_index}, {2, first_dim_size}, {3, outer_dim_size}};
    SendForPush(kPushCmd, kvs, sparse_partitioner_, attrs);
  } else {
    SendForPush(kPushCmd, kvs, round_robin_partitioner_, {});
  }
}

void Worker::PullData(const std::vector<Key> &keys, std::vector<float> *const vals, std::vector<int> *lens, int cmd,
                      int64_t priority) {
  MS_EXCEPTION_IF_NULL(vals);
  KVMessage kvs;
  *kvs.mutable_keys() = {keys.begin(), keys.end()};
  if (embedding_table_ranges_.count(keys[0])) {
    SendForPull(cmd, kvs, broadcast_partitioner_, {}, vals, lens);
  } else {
    SendForPull(cmd, kvs, round_robin_partitioner_, {}, vals, lens);
  }
}

void Worker::LookupIdPartitioner(const EmbeddingTableLookup &send, PartitionEmbeddingMessages *partition,
                                 const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);

  const Key &key = send.key();
  const std::vector<EmbeddingTableShardMetadata> &ranges = *(embedding_table_ranges_[key]);
  partition->resize(ranges.size());

  for (size_t i = 0; i < ranges.size(); i++) {
    const EmbeddingTableShardMetadata &range = ranges[i];
    const auto &begin = range.begin();
    const auto &end = range.end();
    std::unordered_set<int32_t> unique_ids;
    auto &kvs = partition->at(i).second;

    kvs.set_key(key);

    std::for_each(send.keys().begin(), send.keys().end(), [&](int32_t lookup_id) {
      if (lookup_id >= SizeToInt(begin) && lookup_id <= SizeToInt(end)) {
        unique_ids.insert(lookup_id);
      }
    });
    MS_LOG(DEBUG) << "The unique ids size is:" << unique_ids.size();

    for (const auto &lookup_id : unique_ids) {
      kvs.add_keys(lookup_id);
      kvs.add_values(0.0f);
    }

    if (kvs.keys().empty()) {
      partition->at(i).first = false;
    } else {
      partition->at(i).first = true;
    }
  }
}

void Worker::SparsePartitioner(const KVMessage &send, PartitionKVMessages *partition,
                               const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);
  // Init variables
  float *data = const_cast<float *>(send.values().data());

  if (attrs.count(0) == 0 || attrs.count(1) == 0 || attrs.count(2) == 0 || attrs.count(3) == 0) {
    MS_LOG(EXCEPTION) << "Invalid attrs keys";
  }
  auto iter = attrs.find(0);
  size_t grad_index = static_cast<size_t>(iter->second);
  iter = attrs.find(1);
  size_t indice_index = static_cast<size_t>(iter->second);
  iter = attrs.find(2);
  size_t first_dim_size = static_cast<size_t>(iter->second);
  iter = attrs.find(3);
  size_t outer_dim_size = static_cast<size_t>(iter->second);

  int grad_size = send.len()[grad_index];
  int indice_size = send.len()[indice_index];
  int segment_size = grad_size / indice_size;

  int64_t grad_offset = 0;
  int64_t indice_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += send.len()[i];
  }
  for (size_t j = 0; j < indice_index; j++) {
    indice_offset += send.len()[j];
  }

  float *grad_data = data + grad_offset;
  void *indice_data_temp = data + indice_offset;
  int *indice_data = reinterpret_cast<int *>(indice_data_temp);

  // Build the mappings of indice to gradient
  std::vector<std::pair<int, float *>> indice_to_grads;
  for (int i = 0; i < indice_size; i++) {
    int indice = indice_data[i];
    float *grad = grad_data + i * segment_size;
    indice_to_grads.push_back(std::make_pair(indice, grad));
  }

  const Key &key = send.keys()[0];
  const std::vector<EmbeddingTableShardMetadata> &ranges = *(embedding_table_ranges_[key]);
  partition->resize(ranges.size());

  // Construct reduced sparse data for each server
  for (size_t i = 0; i < ranges.size(); i++) {
    const EmbeddingTableShardMetadata &range = ranges[i];
    const auto &begin = range.begin();
    const auto &end = range.end();
    auto &kvs = partition->at(i).second;
    *kvs.mutable_keys() = {send.keys().begin(), send.keys().end()};
    *kvs.mutable_len() = {send.len().begin(), send.len().end()};

    // Prepare the sparse gradient and indice
    std::vector<int> indice_ids;
    std::unordered_set<int> distinct_ids;
    for (int j = 0; j < indice_size; j++) {
      size_t indice = static_cast<size_t>(indice_data[j]);
      if (indice >= begin && indice <= end) {
        indice_ids.push_back(indice);
        distinct_ids.insert(indice);
      }
    }
    size_t indices_size = indice_ids.size();
    if (indices_size > 0) {
      int partition_segment_size = indices_size * segment_size;
      std::vector<float> src_grad_data(partition_segment_size);
      std::vector<int> src_indice_data(indices_size);
      PrepareSparseGradient(begin, end, distinct_ids, indice_to_grads, indice_data, segment_size, src_grad_data.data(),
                            src_indice_data.data());

      // Reduce the sparse gradient and indice
      std::vector<float> new_grad(partition_segment_size);
      std::vector<int> new_indices(indices_size);
      mindspore::kernel::SparseGradient<int> unique_sparse_grad({new_grad.data(), new_indices.data(), indices_size});
      Util::ReduceSparseGradient(src_grad_data.data(), src_indice_data.data(), indices_size, segment_size,
                                 first_dim_size, outer_dim_size, &unique_sparse_grad);

      // Update the length of reduce sparse gradient and indice
      std::vector<int> reduced_lens;
      reduced_lens = {kvs.len().begin(), kvs.len().end()};
      reduced_lens[grad_index] = unique_sparse_grad.indices_size_ * segment_size;
      reduced_lens[indice_index] = unique_sparse_grad.indices_size_;

      // Build the sparse value to be sent
      size_t total_size = std::accumulate(reduced_lens.begin(), reduced_lens.end(), 0, std::plus<int>());
      std::vector<float> reduced_data(total_size, 0);
      BuildSparseValue(reduced_lens, grad_index, indice_index, data, unique_sparse_grad.value_,
                       unique_sparse_grad.indices_, &reduced_data);

      *kvs.mutable_len() = {reduced_lens.begin(), reduced_lens.end()};
      *kvs.mutable_values() = {reduced_data.begin(), reduced_data.end()};
    }

    if (indices_size == 0) {
      std::vector<float> no_keys;
      std::vector<float> no_vals;
      std::vector<float> no_lens;
      no_keys.push_back(key);
      no_vals.push_back(-100);
      *kvs.mutable_values() = {no_vals.begin(), no_vals.end()};
      *kvs.mutable_len() = {no_lens.begin(), no_lens.end()};
    }
    partition->at(i).first = true;
  }
}

void Worker::RoundRobinPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                                   const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);
  partition->resize(server_num_);
  auto keys = send.keys();
  auto values = send.values();
  auto lens = send.len();
  MS_LOG(INFO) << "the key size is:" << send.keys_size() << " the values size is:" << send.values_size()
               << " the lens:" << send.len_size();

  int64_t len;
  Key param_key;
  for (int i = 0; i < send.keys_size(); i++) {
    param_key = keys[i];
    int64_t server_id = key_to_server_id_[param_key];
    if (!partition->at(server_id).first) {
      partition->at(server_id).first = true;
    }

    KVMessage &server_kv_pairs = partition->at(server_id).second;
    server_kv_pairs.add_keys(param_key);
    if (values.empty()) {
      continue;
    }
    len = lens[i];
    int64_t offset = std::accumulate(lens.begin(), lens.begin() + i, 0);
    auto val_begin = values.begin() + offset;
    auto val_end = val_begin + len;
    for (auto it = val_begin; it != val_end; ++it) {
      server_kv_pairs.add_values(*it);
    }
    server_kv_pairs.add_len(len);
  }
}

void Worker::WorkerInitEmbeddingPartitioner(const KVMessage &send, std::vector<std::pair<bool, KVMessage>> *partition,
                                            const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);
  partition->resize(server_num_);
  auto keys = send.keys();
  auto values = send.values();
  auto lens = send.len();

  size_t col_cnt = lens[0] / embedding_row_cnt_[keys[0]];
  const std::vector<EmbeddingTableShardMetadata> &ranges = *(embedding_table_ranges_[keys[0]]);
  for (size_t i = 0; i < ranges.size(); i++) {
    size_t offset_begin = ranges[i].begin() * col_cnt;
    size_t offset_end = (ranges[i].end() + 1) * col_cnt;
    KVMessage kvs;
    *kvs.mutable_keys() = keys;
    *kvs.mutable_values() = {values.begin() + offset_begin, values.begin() + offset_end};
    kvs.add_len(offset_end - offset_begin);
    partition->at(i).first = true;
    partition->at(i).second = kvs;
  }
}
void Worker::UpdateEmbeddingPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                                        const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);
  const float *embedding_vals = send.values().data();
  const int *lookup_ids = send.len().data();
  size_t val_size = send.values_size();
  size_t id_size = send.len_size();
  size_t embedding_dim = val_size / id_size;

  const Key &key = send.keys()[0];
  const std::vector<EmbeddingTableShardMetadata> &ranges = *(embedding_table_ranges_[key]);
  partition->resize(ranges.size());

  for (size_t i = 0; i < ranges.size(); i++) {
    const EmbeddingTableShardMetadata &range = ranges[i];
    const auto &begin = range.begin();
    const auto &end = range.end();
    auto &kvs = partition->at(i).second;
    kvs.add_keys(key);
    for (size_t j = 0; j < id_size; j++) {
      auto lookup_id = static_cast<uint64_t>(lookup_ids[j]);
      if (lookup_id >= begin && lookup_id <= end) {
        kvs.add_keys(lookup_id);
        for (size_t k = 0; k < embedding_dim; k++) {
          kvs.add_values(embedding_vals[j * embedding_dim + k]);
        }
      }
    }

    if (kvs.keys_size() <= 1) {
      partition->at(i).first = false;
    } else {
      partition->at(i).first = true;
    }
  }
}

void Worker::BroadcastPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                                  const std::map<int64_t, int64_t> &attrs) {
  MS_EXCEPTION_IF_NULL(partition);
  partition->resize(server_num_);
  for (int64_t i = 0; i < server_num_; i++) {
    partition->at(i).first = true;
    partition->at(i).second = send;
  }
}

void Worker::SendForPush(int cmd, const KVMessage &send, const KVPartitioner &partitioner,
                         const std::map<int64_t, int64_t> &attrs) {
  PartitionKVMessages messages;
  partitioner(send, &messages, attrs);
  std::vector<uint32_t> rank_ids;
  std::vector<DataPtr> data;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < messages.size(); i++) {
    if (messages.at(i).first) {
      rank_ids.push_back(i);
      std::string kv_data = messages.at(i).second.SerializeAsString();

      std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
      size_t dest_size = kv_data.length();
      int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      data.push_back(res);
      sizes.push_back(kv_data.length());
    }
  }
  worker_node_.Send(core::NodeRole::SERVER, rank_ids, data, sizes, cmd);
}

void Worker::SendForPull(int cmd, const KVMessage &send, const KVPartitioner &partitioner,
                         const std::map<int64_t, int64_t> &attrs, std::vector<float> *vals, std::vector<int> *lens) {
  MS_EXCEPTION_IF_NULL(vals);
  PartitionKVMessages messages;
  partitioner(send, &messages, {});
  std::vector<uint32_t> rank_ids;
  std::vector<DataPtr> data;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < messages.size(); i++) {
    if (messages.at(i).first) {
      rank_ids.push_back(i);
      std::string kv_data = messages.at(i).second.SerializeAsString();

      std::shared_ptr<unsigned char[]> res(new unsigned char[kv_data.length()]);
      size_t dest_size = kv_data.length();
      int ret = memcpy_s(res.get(), dest_size, kv_data.data(), kv_data.length());
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      data.push_back(res);
      sizes.push_back(kv_data.length());
    }
  }
  std::vector<VectorPtr> resp;
  worker_node_.Send(core::NodeRole::SERVER, rank_ids, data, sizes, cmd, &resp);
  vals->clear();
  for (size_t i = 0; i < resp.size(); ++i) {
    KVMessage message;
    message.ParseFromArray(resp.at(i)->data(), resp.at(i)->size());
    std::copy(message.values().begin(), message.values().end(), std::back_inserter(*vals));

    if (lens) {
      lens->clear();
      std::copy(message.len().begin(), message.len().end(), std::back_inserter(*lens));
    }
  }
}
}  // namespace ps
}  // namespace mindspore
