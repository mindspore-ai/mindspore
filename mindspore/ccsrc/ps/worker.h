/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_WORKER_H_
#define MINDSPORE_CCSRC_PS_WORKER_H_

#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <algorithm>
#include <map>
#include "ps/ps.h"
#include "utils/log_adapter.h"
#include "ir/tensor.h"
#include "ps/util.h"
#include "ps/common.h"
#include "ps/worker_proxy.h"
#include "utils/shape_utils.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"

namespace mindspore {
namespace ps {
template <typename T>
class Worker {
 public:
  static Worker &GetInstance() {
    static Worker instance;
    return instance;
  }

  void Run();
  void Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const ShapeVector &sizes);
  void Pull(const size_t key, void *dev_addr, const size_t size);
  size_t SetParamKey(const std::string &param_name);
  size_t GetParamKey(const std::string &param_name);
  void SetParamInitInServer(const std::string &param_name, bool init_in_server);
  bool GetParamInitInServer(const std::string &param_name);
  void SetKeyOptimId(size_t key, const std::string &optimizer_name);
  void SetOptimInputShapes(size_t key, const ShapeVector &shape);
  void AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count);
  void InitPSEmbeddingTable(const std::vector<size_t> &keys, std::vector<T> shapes, const ShapeVector &sizes);
  void InitPSParamAndOptim(const AnfNodePtr &input_node, const tensor::TensorPtr &tensor);
  void DoPSEmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                           const ::ps::SArray<int> &lens, ::ps::SArray<T> *lookup_result, int64_t cmd);
  void UpdateEmbeddingTable(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                            const ::ps::SArray<T> &vals);
  bool running() { return running_; }
  void Finalize();

 private:
  Worker() : kv_worker_(nullptr), running_(false), key_cnt_(0) {}
  ~Worker() = default;
  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;

  bool IsKeyInit(const size_t key);
  void InitPSOptimId(const size_t param_key);
  void InitPSOptimInputShapes(const size_t key);
  void InitPSParamData(const std::vector<size_t> &keys, void *origin_addr, size_t size);
  static void EmbeddingLookupIdSlicer(const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &ranges,
                                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced) {}

  std::shared_ptr<WorkerProxy<T>> kv_worker_;
  bool running_;
  size_t key_cnt_;
  std::map<std::string, size_t> param_to_key_;
  std::map<size_t, bool> init_keys_;
  std::map<size_t, int64_t> key_to_optimId_;
  std::map<size_t, std::vector<ShapeVector>> key_to_optim_shapes_;
  std::map<std::string, bool> param_to_init_in_server_;
};

template <typename T>
void Worker<T>::Run() {
  if (running_) {
    MS_LOG(INFO) << "'Worker is already running.";
    return;
  }
  MS_LOG(INFO) << "Worker starts connecting to scheduler and server...";
  ::ps::Start(0);
  MS_LOG(INFO) << "Worker connected successfully.";
  if (!::ps::IsWorker()) {
    MS_LOG(EXCEPTION) << "The role is not worker.";
  }
  kv_worker_ = std::make_shared<WorkerProxy<T>>(0, 0, 1, 2);
  running_ = true;
}

template <typename T>
void Worker<T>::Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const ShapeVector &sizes) {
  if (keys.size() == 0) {
    MS_LOG(EXCEPTION) << "key size should be greater than zero";
  }
  if (key_to_optimId_.count(keys[0]) == 0) {
    MS_LOG(EXCEPTION) << "no optim id found for key" << keys[0];
  }
  Key key = keys[0];
  int64_t optim_id = key_to_optimId_[key];
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
  ::ps::SArray<T> total_buffer(total_size, 0);
  size_t offset = 0;
  size_t dst_size = 0;
  size_t src_size = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    void *dst_data = total_buffer.data() + offset / sizeof(T);
    void *src_data = reinterpret_cast<void *>(addrs[i]);
    MS_EXCEPTION_IF_NULL(dst_data);
    MS_EXCEPTION_IF_NULL(src_data);
    dst_size = sizes[i] * sizeof(T);
    src_size = sizes[i] * sizeof(T);
    auto ret = memcpy_s(dst_data, dst_size, src_data, src_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }
    offset += sizes[i] * sizeof(T);
  }

  while (!kv_worker_->IsReadyForPush(keys[0])) {
    continue;
  }
  std::vector<int> sizes_int;
  (void)std::transform(sizes.begin(), sizes.end(), std::back_inserter(sizes_int),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (!is_sparse) {
    kv_worker_->PushData(::ps::SArray<::ps::Key>(keys), total_buffer, ::ps::SArray<int>(sizes_int));
  } else {
    std::vector<int64_t> &var_shape = key_to_optim_shapes_[key][0];
    int64_t first_dim_size = var_shape[0];
    int64_t outer_dim_size = std::accumulate(var_shape.begin() + 1, var_shape.end(), 1, std::multiplies<int64_t>());
    kv_worker_->PushSparseData(::ps::SArray<::ps::Key>(keys), total_buffer, ::ps::SArray<int>(sizes_int), grad_index,
                               indice_index, first_dim_size, outer_dim_size);
  }
}

template <typename T>
void Worker<T>::Pull(const size_t key, void *dev_addr, const size_t size) {
  MS_EXCEPTION_IF_NULL(dev_addr);
  ::ps::SArray<T> variables(size / sizeof(T), 0);
  while (!kv_worker_->IsReadyForPull(key)) {
    continue;
  }
  kv_worker_->PullData({key}, &variables);
  size_t dst_size = size;
  size_t src_size = size;
  auto ret = memcpy_s(dev_addr, dst_size, variables.data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
}

template <typename T>
void Worker<T>::DoPSEmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                                    const ::ps::SArray<int> &lens, ::ps::SArray<T> *lookup_result, int64_t cmd) {
  MS_EXCEPTION_IF_NULL(lookup_result);
  kv_worker_->EmbeddingLookup(keys, lookup_ids, lens, lookup_result, cmd);
}

template <typename T>
void Worker<T>::UpdateEmbeddingTable(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                                     const ::ps::SArray<T> &vals) {
  kv_worker_->UpdateEmbeddingTable(keys, lookup_ids, vals);
}

template <typename T>
void Worker<T>::Finalize() {
  if (running_) {
    MS_LOG(INFO) << "Worker starts finalizing...";
    kv_worker_->Finalize();
    kv_worker_.reset();
    running_ = false;
    MS_LOG(INFO) << "Worker finalized successfully.";
  }
}

template <typename T>
void Worker<T>::InitPSParamData(const std::vector<size_t> &keys, void *origin_addr, size_t size) {
  MS_EXCEPTION_IF_NULL(origin_addr);
  ::ps::SArray<T> addr(reinterpret_cast<T *>(origin_addr), size / sizeof(T));
  ::ps::SArray<::ps::Key> key(keys);
  ::ps::SArray<int> lens;
  lens.push_back(addr.size());
  kv_worker_->PushData(key, addr, lens, kInitWeightsCmd);
  init_keys_[key[0]] = true;
}

template <typename T>
void Worker<T>::SetOptimInputShapes(size_t key, const ShapeVector &shape) {
  if (key_to_optim_shapes_.find(key) == key_to_optim_shapes_.end()) {
    key_to_optim_shapes_[key] = {shape};
  } else {
    key_to_optim_shapes_[key].push_back(shape);
  }
}

template <typename T>
void Worker<T>::InitPSOptimInputShapes(const size_t key) {
  ::ps::SArray<::ps::Key> keys;
  ::ps::SArray<int> shape_len;
  ::ps::SArray<T> all_shape;
  std::vector<ShapeVector> shapes = key_to_optim_shapes_[key];
  for (auto shape : shapes) {
    keys.push_back(key);
    if (shape.size() == 0) {
      shape_len.push_back(1);
      all_shape.push_back(1);
    } else {
      shape_len.push_back(SizeToLong(shape.size()));
      for (auto dim : shape) {
        all_shape.push_back(static_cast<T>(dim));
      }
    }
  }
  MS_LOG(INFO) << "keys:" << keys;
  MS_LOG(INFO) << "shape_len:" << shape_len;
  MS_LOG(INFO) << "all_shape:" << all_shape;
  if (!init_keys_[key]) {
    init_keys_[key] = true;
  }
  kv_worker_->PushData(keys, all_shape, shape_len, kInitOptimInputsShapeCmd);
}

template <typename T>
bool Worker<T>::IsKeyInit(const size_t key) {
  if (init_keys_.find(key) == init_keys_.end() || !init_keys_[key]) {
    return false;
  }
  return true;
}

template <typename T>
size_t Worker<T>::SetParamKey(const std::string &param_name) {
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

template <typename T>
void Worker<T>::SetParamInitInServer(const std::string &param_name, bool init_in_server) {
  MS_LOG(INFO) << "Set parameter " << param_name << " init_in_server:" << init_in_server;
  param_to_init_in_server_[param_name] = init_in_server;
}

template <typename T>
bool Worker<T>::GetParamInitInServer(const std::string &param_name) {
  if (param_to_init_in_server_.count(param_name) == 0) {
    return false;
  }
  return param_to_init_in_server_[param_name];
}

template <typename T>
size_t Worker<T>::GetParamKey(const std::string &param_name) {
  size_t key = kInvalidKey;
  if (param_to_key_.find(param_name) != param_to_key_.end()) {
    key = param_to_key_[param_name];
    MS_LOG(DEBUG) << "Get key of parameter " << param_name << " key is " << key;
  }
  return key;
}

template <typename T>
void Worker<T>::SetKeyOptimId(size_t key, const std::string &optimizer_name) {
  key_to_optimId_[key] = Util::optimizer_id(optimizer_name);
}

template <typename T>
void Worker<T>::InitPSOptimId(const size_t param_key) {
  if (key_to_optimId_.count(param_key) == 0) {
    MS_LOG(EXCEPTION) << "Can't find optimizer id of parameter key " << param_key;
  }
  int64_t optim_id = key_to_optimId_[param_key];

  ::ps::SArray<::ps::Key> keys = {param_key};
  ::ps::SArray<T> optim_id_vals = {static_cast<T>(optim_id)};
  ::ps::SArray<int> optim_id_lens = {optim_id_vals.size()};
  kv_worker_->PushData(keys, optim_id_vals, optim_id_lens, kInitWeightToOptimIdCmd);
}

template <typename T>
void Worker<T>::InitPSEmbeddingTable(const std::vector<size_t> &keys, std::vector<T> shapes, const ShapeVector &sizes) {
  bool has_init = IsKeyInit(keys[0]);
  if (has_init) {
    MS_LOG(DEBUG) << "The key embedding table of key " << keys[0] << " is initialized.";
    return;
  }
  ::ps::SArray<T> shapes_val;
  for (auto dim : shapes) {
    shapes_val.push_back(dim);
  }
  std::vector<int> sizes_int;
  (void)std::transform(sizes.begin(), sizes.end(), std::back_inserter(sizes_int),
                       [](const int64_t &value) { return static_cast<int>(value); });
  kv_worker_->Wait(
    kv_worker_->InitEmbeddingTable(::ps::SArray<::ps::Key>(keys), shapes_val, ::ps::SArray<int>(sizes_int)));
}

template <typename T>
void Worker<T>::InitPSParamAndOptim(const AnfNodePtr &input_node, const tensor::TensorPtr &tensor) {
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
    MS_LOG(INFO) << "Init paramter and optimizer in parameter server side for " << param_name
                 << ", whether init in server: " << init_in_server;
    kv_worker_->AddKeyToServerId(param_key);
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

template <typename T>
void Worker<T>::AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count) {
  bool has_init = IsKeyInit(key);
  if (has_init) {
    return;
  }
  kv_worker_->AddEmbeddingTable(key, row_count);
}

static Worker<float> &worker = Worker<float>::GetInstance();
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_WORKER_H_
