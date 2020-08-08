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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_H_

#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include "ps/ps.h"
#include "utils/log_adapter.h"
#include "ir/tensor.h"
#include "frontend/parallel/ps/util.h"
#include "frontend/parallel/ps/common.h"
#include "frontend/parallel/ps/worker_proxy.h"

namespace mindspore {
namespace parallel {
namespace ps {
template <typename T>
class Worker {
 public:
  static Worker &GetInstance() {
    static Worker instance;
    return instance;
  }

  void Run();
  void Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const std::vector<int> &sizes);
  void Pull(const size_t key, void *dev_addr, const size_t size);
  size_t SetParamKey(const std::string &param_name);
  void SetParamInitInServer(const std::string &param_name, bool init_in_server);
  bool GetParamInitInServer(const std::string &param_name);
  void SetKeyOptimId(size_t key, const std::string &optimizer_name);
  void SetOptimInputShapes(size_t key, const std::vector<int> &shape);
  void AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count);
  void InitPSEmbeddingTable(const std::vector<size_t> &keys, std::vector<size_t> shapes, const std::vector<int> &sizes);
  void InitPSParamAndOptim(const std::string &param_name, tensor::TensorPtr tensor);
  void DoPSEmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                           const ::ps::SArray<int> &lens, ::ps::SArray<T> *lookup_result, int cmd);
  void Finalize();

 private:
  Worker() : kv_worker_(nullptr), running_(false), key_cnt_(0) {}
  ~Worker() = default;
  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;

  bool IsKeyInit(const size_t key);
  size_t GetParamKey(const std::string &param_name);
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
  std::map<size_t, int> key_to_optimId_;
  std::map<size_t, std::vector<std::vector<int>>> key_to_optim_shapes_;
  std::map<std::string, bool> param_to_init_in_server_;
};

template <typename T>
void Worker<T>::Run() {
  if (running_) {
    MS_LOG(INFO) << "'Worker is already running.";
    return;
  }
  ::ps::Start(0);
  if (!::ps::IsWorker()) {
    MS_LOG(EXCEPTION) << "The role is not worker.";
  }
  kv_worker_ = std::make_shared<WorkerProxy<T>>(0, 0, 1);
  running_ = true;
}

template <typename T>
void Worker<T>::Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const std::vector<int> &sizes) {
  size_t total_size = 0;
  for (auto size : sizes) {
    total_size += size;
  }
  ::ps::SArray<T> total_buffer(total_size, 0);
  size_t offset = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    auto ret = memcpy_s(total_buffer.data() + offset / sizeof(T), sizes[i] * sizeof(T),
                        reinterpret_cast<void *>(addrs[i]), sizes[i] * sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    offset += sizes[i] * sizeof(T);
  }
  while (!kv_worker_->IsReadyForPush(keys[0])) {
    continue;
  }
  kv_worker_->PushData(::ps::SArray<::ps::Key>(keys), total_buffer, ::ps::SArray<int>(sizes));
}

template <typename T>
void Worker<T>::Pull(const size_t key, void *dev_addr, const size_t size) {
  ::ps::SArray<T> variables(size / sizeof(T), 0);
  while (!kv_worker_->IsReadyForPull(key)) {
    continue;
  }
  kv_worker_->Wait(kv_worker_->ZPull({key}, &variables));
  auto ret = memcpy_s(dev_addr, size, variables.data(), size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
}

template <typename T>
void Worker<T>::DoPSEmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                                    const ::ps::SArray<int> &lens, ::ps::SArray<T> *lookup_result, int cmd) {
  kv_worker_->EmbeddingLookup(keys, lookup_ids, lens, lookup_result, cmd);
}

template <typename T>
void Worker<T>::Finalize() {
  if (running_) {
    kv_worker_->Finalize();
    kv_worker_.reset();
    running_ = false;
  }
}

template <typename T>
void Worker<T>::InitPSParamData(const std::vector<size_t> &keys, void *origin_addr, size_t size) {
  ::ps::SArray<T> addr(reinterpret_cast<T *>(origin_addr), size / sizeof(T));
  ::ps::SArray<::ps::Key> key(keys);
  ::ps::SArray<int> lens;
  lens.push_back(addr.size());
  kv_worker_->Wait(kv_worker_->ZPush(key, addr, lens, kInitWeightsCmd));
  init_keys_[key[0]] = true;
}

template <typename T>
void Worker<T>::SetOptimInputShapes(size_t key, const std::vector<int> &shape) {
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
  std::vector<std::vector<int>> shapes = key_to_optim_shapes_[key];
  for (auto shape : shapes) {
    keys.push_back(key);
    if (shape.size() == 0) {
      shape_len.push_back(1);
      all_shape.push_back(1);
    } else {
      shape_len.push_back(SizeToInt(shape.size()));
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
    MS_LOG(INFO) << "Get key of parameter " << param_name << " key is " << key;
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
  int optim_id = key_to_optimId_[param_key];

  ::ps::SArray<::ps::Key> keys = {param_key};
  ::ps::SArray<T> optim_id_vals = {static_cast<T>(optim_id)};
  ::ps::SArray<int> optim_id_lens = {optim_id_vals.size()};
  kv_worker_->PushData(keys, optim_id_vals, optim_id_lens, kInitWeightToOptimIdCmd);
}

template <typename T>
void Worker<T>::InitPSEmbeddingTable(const std::vector<size_t> &keys, std::vector<size_t> shapes,
                                     const std::vector<int> &sizes) {
  bool has_init = IsKeyInit(keys[0]);
  if (has_init) {
    MS_LOG(DEBUG) << "The key embedding table of key " << keys[0] << " is initialized.";
    return;
  }
  ::ps::SArray<T> shapes_val;
  for (auto dim : shapes) {
    shapes_val.push_back(static_cast<T>(dim));
  }
  kv_worker_->Wait(kv_worker_->InitEmbeddingTable(::ps::SArray<::ps::Key>(keys), shapes_val, ::ps::SArray<int>(sizes)));
}

template <typename T>
// Initialize parameters and optimizer kernels of Parameter Server.
void Worker<T>::InitPSParamAndOptim(const std::string &param_name, tensor::TensorPtr tensor) {
  void *param_data = tensor->data_c();
  size_t param_size = LongToSize(tensor->data().nbytes());
  std::vector<int> param_shape = tensor->shape_c();

  size_t param_key = GetParamKey(param_name);
  if (param_key == kInvalidKey) {
    MS_LOG(INFO) << "Parameter " << param_name << " has no key assigned.";
    return;
  }
  bool init_in_server = false;
  std::vector<int> shape_init_in_server = {1};
  if (param_shape == shape_init_in_server) {
    init_in_server = true;
  }
  SetParamInitInServer(param_name, init_in_server);
  bool init = IsKeyInit(param_key);
  if (!init) {
    MS_LOG(INFO) << "Init paramter and optimizer in parameter server side for " << param_name
                 << ", whether init in server: " << init_in_server;
    if (!init_in_server) {
      InitPSParamData({param_key}, param_data, param_size);
    }
    InitPSOptimId(param_key);
    InitPSOptimInputShapes(param_key);
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

}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_H_
