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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_PROXY_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_PROXY_H_

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include "ps/ps.h"
#include "frontend/parallel/ps/util.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace parallel {
namespace ps {
template <typename T>
class WorkerProxy : public ::ps::KVWorker<T> {
 public:
  using Worker = ::ps::KVWorker<T>;
  using Callback = std::function<void()>;
  using SlicedKVs = std::vector<std::pair<bool, ::ps::KVPairs<T>>>;
  using Slicer = std::function<void(int ts, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &ranges,
                                    SlicedKVs *sliced, const std::map<int, int> &attrs)>;
  using ::ps::SimpleApp::obj_;
  explicit WorkerProxy(int app_id, int customer_id, int lookup_customer_id, int general_customer_id)
      : Worker(app_id, customer_id) {
    server_num_ = ::ps::NumServers();
    Util::SetRankId(::ps::MyRank());
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    using std::placeholders::_5;
    lookup_customer_ = std::unique_ptr<::ps::Customer>(
      new ::ps::Customer(app_id, lookup_customer_id, std::bind(&WorkerProxy<T>::ProcessLookupResult, this, _1)));
    general_customer_ = std::unique_ptr<::ps::Customer>(
      new ::ps::Customer(app_id, general_customer_id, std::bind(&WorkerProxy<T>::ProcessResponse, this, _1)));
    lookup_slicer_ = std::bind(&WorkerProxy<T>::LookupIdSlicer, this, _1, _2, _3, _4, _5);
    sparse_slicer_ = std::bind(&WorkerProxy<T>::SparseSlicer, this, _1, _2, _3, _4, _5);
    broadcast_slicer_ = std::bind(&WorkerProxy<T>::BroadcastSlicer, this, _1, _2, _3, _4, _5);
    round_robin_slicer_ = std::bind(&WorkerProxy<T>::RoundRobinSlicer, this, _1, _2, _3, _4, _5);
    worker_init_embedding_slicer_ = std::bind(&WorkerProxy<T>::WorkerInitEmbeddingSlicer, this, _1, _2, _3, _4, _5);
  }
  ~WorkerProxy() override = default;

  void AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count);
  void AddKeyToServerId(const ::ps::Key &key);
  void EmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                       const ::ps::SArray<int> &lens, ::ps::SArray<T> *outs, int cmd = 0, const Callback &cb = nullptr,
                       int priority = 0);
  int InitEmbeddingTable(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                         const ::ps::SArray<int> &lens = {}, const Callback &cb = nullptr, int priority = 0);
  bool IsReadyForPush(const Key &key);
  bool IsReadyForPull(const Key &key);
  void PushData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals, const ::ps::SArray<int> &lens = {},
                int cmd = 0, int priority = 0);
  void PushSparseData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals, const ::ps::SArray<int> &lens,
                      size_t grad_index, size_t indice_index, size_t first_dim_size, size_t outer_dim_size);
  void PullData(const ::ps::SArray<::ps::Key> &keys, ::ps::SArray<T> *vals, ::ps::SArray<int> *lens = nullptr,
                int cmd = 0, int priority = 0);
  void Finalize();

 private:
  template <typename C>
  int AddLookupCB(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids, C *vals, int cmd,
                  const Callback &cb);
  int AddGeneralRspCB(const ::ps::SArray<::ps::Key> &keys, ::ps::SArray<T> *vals, ::ps::SArray<int> *lens, int cmd,
                      const Callback &cb);
  void LookupIdSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced, const std::map<int, int> &attrs);
  void SparseSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                    std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced, const std::map<int, int> &attrs);
  void BroadcastSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                       std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced, const std::map<int, int> &attrs);
  void RoundRobinSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                        std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced, const std::map<int, int> &attrs);
  void WorkerInitEmbeddingSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                 std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                 const std::map<int, int> &attrs);
  void ProcessLookupResult(const ::ps::Message &msg);
  void ProcessResponse(const ::ps::Message &msg);
  void Send(::ps::Customer *customer, int timestamp, bool push, bool pull, int cmd, const ::ps::KVPairs<T> &kvs,
            const Slicer &slicer, std::map<int, int> attrs = {});
  void AddKeyByHashMod(const ::ps::Key &key);

  void PrepareSparseGradient(const size_t begin, const size_t end, const std::unordered_set<int> &distinct_ids,
                             const std::vector<std::pair<int, T *>> &indice_to_grad, const int *all_indice,
                             const size_t segment_size, T *gradient, int *indice);
  void BuildSparseValue(const ::ps::SArray<int> &lengths, const size_t grad_index, const size_t indice_index,
                        const T *original_data, const T *grads, int *indices, ::ps::SArray<T> *reduced_data);

  int server_num_;
  std::unique_ptr<::ps::Customer> lookup_customer_;
  std::unique_ptr<::ps::Customer> general_customer_;
  std::unordered_map<::ps::Key, std::shared_ptr<std::vector<::ps::Range>>> embedding_table_ranges_;
  std::unordered_map<int, std::vector<::ps::KVPairs<T>>> lookup_results_;
  std::unordered_map<int, std::map<int, ::ps::KVPairs<T>>> gathered_response_;
  std::mutex mutex_;
  Slicer lookup_slicer_;
  Slicer sparse_slicer_;
  Slicer broadcast_slicer_;
  Slicer round_robin_slicer_;
  Slicer worker_init_embedding_slicer_;
  std::unordered_map<int, Callback> lookup_callbacks_;
  std::unordered_map<int, Callback> general_callbacks_;
  std::unordered_map<int, int> expected_result_count_;
  std::unordered_map<::ps::Key, int> key_to_server_id_;
  std::unordered_map<::ps::Key, size_t> embedding_row_cnt_;
};

template <typename T>
void WorkerProxy<T>::AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count) {
  uint64_t begin = 0;
  uint64_t end = 0;
  for (int i = 0; i < server_num_; i++) {
    int local_row_cnt = Util::LocalShard(row_count, i, server_num_);
    if (i == 0) {
      end = local_row_cnt - 1;
    } else {
      begin = end + 1;
      end += local_row_cnt;
    }
    ::ps::Range range(begin, end);
    if (embedding_table_ranges_.count(key) == 0) {
      embedding_table_ranges_[key] = std::make_shared<std::vector<::ps::Range>>();
    }
    embedding_table_ranges_[key]->push_back(range);
  }
  embedding_row_cnt_[key] = row_count;
}

template <typename T>
void WorkerProxy<T>::AddKeyByHashMod(const ::ps::Key &key) {
  if (server_num_ == 0) {
    MS_LOG(EXCEPTION) << "Server number is invalid:0";
  }
  key_to_server_id_[key] = static_cast<int>(key % server_num_);
  MS_LOG(INFO) << "The server id of key " << key << " is " << key_to_server_id_[key];
}

template <typename T>
void WorkerProxy<T>::AddKeyToServerId(const ::ps::Key &key) {
  AddKeyByHashMod(key);
}

template <typename T>
void WorkerProxy<T>::EmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                                     const ::ps::SArray<int> &lens, ::ps::SArray<T> *outs, int cmd, const Callback &cb,
                                     int priority) {
  int ts = AddLookupCB(keys, lookup_ids, outs, cmd, cb);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.lens = lookup_ids;
  kvs.priority = priority;
  expected_result_count_[ts] = 0;
  Send(lookup_customer_.get(), ts, true, true, cmd, kvs, lookup_slicer_);
  int expect_rt_count = expected_result_count_[ts];
  lookup_customer_->AddResponse(ts, server_num_ - expect_rt_count);
  lookup_customer_->WaitRequest(ts);
  expected_result_count_.erase(ts);
}

template <typename T>
int WorkerProxy<T>::InitEmbeddingTable(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                                       const ::ps::SArray<int> &lens, const Callback &cb, int priority) {
  int ts = obj_->NewRequest(::ps::kServerGroup);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.vals = vals;
  kvs.lens = lens;
  kvs.priority = priority;
  Send(obj_, ts, true, false, kInitEmbeddingsCmd, kvs, broadcast_slicer_);
  return ts;
}

template <typename T>
bool WorkerProxy<T>::IsReadyForPush(const Key &key) {
  ::ps::SArray<T> result(1, 0);
  PullData({key}, &result, nullptr, kCheckReadyForPushCmd);
  if (result[0] > 0) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
bool WorkerProxy<T>::IsReadyForPull(const Key &key) {
  ::ps::SArray<T> result(1, 0);
  PullData({key}, &result, nullptr, kCheckReadyForPullCmd);
  if (result[0] > 0) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
void WorkerProxy<T>::PushData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                              const ::ps::SArray<int> &lens, int cmd, int priority) {
  int ts = AddGeneralRspCB(keys, nullptr, nullptr, cmd, nullptr);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.vals = vals;
  kvs.lens = lens;
  kvs.priority = priority;
  if (embedding_table_ranges_.count(keys[0])) {
    if (cmd == kInitWeightsCmd) {
      Send(general_customer_.get(), ts, true, false, cmd, kvs, worker_init_embedding_slicer_);
    } else {
      Send(general_customer_.get(), ts, true, false, cmd, kvs, broadcast_slicer_);
    }
  } else {
    Send(general_customer_.get(), ts, true, false, cmd, kvs, round_robin_slicer_);
  }
  if (expected_result_count_[ts] < server_num_) {
    general_customer_->AddResponse(ts, server_num_ - expected_result_count_[ts]);
  }
  general_customer_->WaitRequest(ts);
}

template <typename T>
void WorkerProxy<T>::PushSparseData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                                    const ::ps::SArray<int> &lens, size_t grad_index, size_t indice_index,
                                    size_t first_dim_size, size_t outer_dim_size) {
  int ts = AddGeneralRspCB(keys, nullptr, nullptr, 0, nullptr);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.vals = vals;
  kvs.lens = lens;
  int cmd = 0;
  if (embedding_table_ranges_.count(keys[0])) {
    std::map<int, int> attrs{{0, grad_index}, {1, indice_index}, {2, first_dim_size}, {3, outer_dim_size}};
    Send(general_customer_.get(), ts, true, false, cmd, kvs, sparse_slicer_, attrs);
  } else {
    Send(general_customer_.get(), ts, true, false, cmd, kvs, round_robin_slicer_);
  }
  if (expected_result_count_[ts] < server_num_) {
    general_customer_->AddResponse(ts, server_num_ - expected_result_count_[ts]);
  }
  general_customer_->WaitRequest(ts);
}

template <typename T>
void WorkerProxy<T>::PullData(const ::ps::SArray<::ps::Key> &keys, ::ps::SArray<T> *vals, ::ps::SArray<int> *lens,
                              int cmd, int priority) {
  int ts = AddGeneralRspCB(keys, vals, lens, cmd, nullptr);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.priority = priority;
  if (embedding_table_ranges_.count(keys[0])) {
    Send(general_customer_.get(), ts, false, true, cmd, kvs, broadcast_slicer_);
  } else {
    Send(general_customer_.get(), ts, false, true, cmd, kvs, round_robin_slicer_);
  }
  if (expected_result_count_[ts] < server_num_) {
    general_customer_->AddResponse(ts, server_num_ - expected_result_count_[ts]);
  }
  general_customer_->WaitRequest(ts);
}

template <typename T>
void WorkerProxy<T>::Finalize() {
  int ts = obj_->NewRequest(::ps::kServerGroup);
  ::ps::KVPairs<T> kvs;
  kvs.keys.push_back(0);
  kvs.vals.push_back(0.0f);
  Send(obj_, ts, true, false, kFinalizeCmd, kvs, broadcast_slicer_);
  obj_->WaitRequest(ts);
  ::ps::Finalize(0, true);
}

template <typename T>
template <typename C>
int WorkerProxy<T>::AddLookupCB(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                                C *lookup_result, int cmd, const Callback &cb) {
  int ts = lookup_customer_->NewRequest(::ps::kServerGroup);
  const auto &callback = [this, ts, keys, lookup_ids, lookup_result, cb]() mutable {
    mutex_.lock();
    auto &kvs = lookup_results_[ts];
    mutex_.unlock();

    std::unordered_map<Key, std::shared_ptr<std::pair<T *, int>>> id_addr_map;
    for (const auto &s : kvs) {
      int offset = 0;
      int len = s.vals.size() / s.keys.size();
      for (size_t i = 0; i < s.keys.size(); i++) {
        const Key &key = s.keys[i];
        T *addr = s.vals.data() + offset;
        offset += len;
        id_addr_map[key] = std::make_shared<std::pair<T *, int>>(std::make_pair(addr, len));
      }
    }

    T *result_addr = lookup_result->data();
    int offset = 0;
    for (size_t i = 0; i < lookup_ids.size(); i++) {
      auto &pair = id_addr_map[static_cast<Key>(lookup_ids[i])];
      int size = pair->second * sizeof(T);
      auto ret = memcpy_s(result_addr + offset, size, pair->first, size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      }
      offset += pair->second;
    }

    mutex_.lock();
    lookup_results_.erase(ts);
    mutex_.unlock();
    if (cb) cb();
  };
  lookup_callbacks_[ts] = callback;
  return ts;
}

template <typename T>
int WorkerProxy<T>::AddGeneralRspCB(const ::ps::SArray<::ps::Key> &keys, ::ps::SArray<T> *vals, ::ps::SArray<int> *lens,
                                    int cmd, const Callback &cb) {
  int ts = general_customer_->NewRequest(::ps::kServerGroup);
  const auto &callback = [this, ts, keys, vals, lens, cb]() mutable {
    mutex_.lock();
    std::map<int, ::ps::KVPairs<T>> server_kvs = gathered_response_[ts];
    mutex_.unlock();

    vals->clear();
    for (auto kvs : server_kvs) {
      for (auto val : kvs.second.vals) {
        vals->push_back(val);
      }
      if (lens) {
        for (auto len : kvs.second.lens) {
          lens->push_back(len);
        }
      }
    }

    mutex_.lock();
    gathered_response_.erase(ts);
    mutex_.unlock();
    if (cb) {
      cb();
    }
  };
  general_callbacks_[ts] = callback;
  return ts;
}

template <typename T>
void WorkerProxy<T>::LookupIdSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                    std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                    const std::map<int, int> &attrs) {
  int *lookup_ids = send.lens.data();
  size_t id_size = send.lens.size();

  const Key &key = send.keys[0];
  const std::vector<::ps::Range> &ranges = *(embedding_table_ranges_[key]);
  sliced->resize(ranges.size());

  for (size_t i = 0; i < ranges.size(); i++) {
    const ::ps::Range &range = ranges[i];
    const auto &begin = range.begin();
    const auto &end = range.end();
    std::unordered_set<int> unique_ids;
    auto &kvs = sliced->at(i).second;

    kvs.keys.push_back(key);
    kvs.vals.push_back(0.0f);

    for (size_t j = 0; j < id_size; j++) {
      auto lookup_id = static_cast<uint64_t>(lookup_ids[j]);
      if (lookup_id >= begin && lookup_id <= end) {
        unique_ids.insert(lookup_id);
      }
    }
    for (const auto &lookup_id : unique_ids) {
      kvs.keys.push_back(lookup_id);
      kvs.vals.push_back(0.0f);
    }

    if (kvs.keys.size() <= 1) {
      sliced->at(i).first = false;
    } else {
      sliced->at(i).first = true;
      expected_result_count_[timestamp] += 1;
    }
  }
}

template <typename T>
void WorkerProxy<T>::SparseSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                  std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                  const std::map<int, int> &attrs) {
  // Init variables
  T *data = send.vals.data();

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

  int grad_size = send.lens[grad_index];
  int indice_size = send.lens[indice_index];
  int segment_size = grad_size / indice_size;

  int grad_offset = 0;
  int indice_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += send.lens[i];
  }
  for (size_t j = 0; j < indice_index; j++) {
    indice_offset += send.lens[j];
  }

  T *grad_data = data + grad_offset;
  int *indice_data = reinterpret_cast<int *>(data) + indice_offset;

  // Build the mappings of indice to gradient
  std::vector<std::pair<int, T *>> indice_to_grads;
  for (int i = 0; i < indice_size; i++) {
    int indice = indice_data[i];
    T *grad = grad_data + i * segment_size;
    indice_to_grads.push_back(std::make_pair(indice, grad));
  }

  const Key &key = send.keys[0];
  const std::vector<::ps::Range> &ranges = *(embedding_table_ranges_[key]);
  sliced->resize(ranges.size());

  // Construct reduced sparse data for each server
  for (size_t i = 0; i < ranges.size(); i++) {
    const ::ps::Range &range = ranges[i];
    const auto &begin = range.begin();
    const auto &end = range.end();
    auto &kvs = sliced->at(i).second;
    kvs.keys = send.keys;
    kvs.lens = send.lens;

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
      int slice_segment_size = indices_size * segment_size;
      T *src_grad_data = new T[slice_segment_size];
      int *src_indice_data = new int[indices_size];
      PrepareSparseGradient(begin, end, distinct_ids, indice_to_grads, indice_data, segment_size, src_grad_data,
                            src_indice_data);

      // Reduce the sparse gradient and indice
      T *new_grad = new T[slice_segment_size];
      int *new_indices = new int[indices_size];
      mindspore::kernel::SparseGradient<int> unique_sparse_grad({new_grad, new_indices, indices_size});
      Util::ReduceSparseGradient(src_grad_data, src_indice_data, indices_size, segment_size, first_dim_size,
                                 outer_dim_size, &unique_sparse_grad);

      // Update the length of reduce sparse gradient and indice
      ::ps::SArray<int> reduced_lens;
      reduced_lens.CopyFrom(kvs.lens);
      reduced_lens[grad_index] = unique_sparse_grad.indices_size_ * segment_size;
      reduced_lens[indice_index] = unique_sparse_grad.indices_size_;

      // Build the sparse value to be sent
      size_t total_size = 0;
      for (auto size : reduced_lens) {
        total_size += size;
      }
      ::ps::SArray<T> reduced_data(total_size, 0);
      BuildSparseValue(reduced_lens, grad_index, indice_index, data, unique_sparse_grad.value_,
                       unique_sparse_grad.indices_, &reduced_data);

      kvs.lens = reduced_lens;
      kvs.vals = reduced_data;

      delete[] src_grad_data;
      delete[] src_indice_data;
      delete[] new_grad;
      delete[] new_indices;
    }

    if (indices_size <= 0) {
      ::ps::SArray<T> no_keys;
      ::ps::SArray<T> no_vals;
      ::ps::SArray<T> no_lens;
      no_keys.push_back(key);
      no_vals.push_back(-100);
      kvs.vals = no_vals;
      kvs.lens = no_lens;
    }
    sliced->at(i).first = true;
    expected_result_count_[timestamp] += 1;
  }
}

template <typename T>
void WorkerProxy<T>::PrepareSparseGradient(const size_t begin, const size_t end,
                                           const std::unordered_set<int> &distinct_ids,
                                           const std::vector<std::pair<int, T *>> &indice_to_grads,
                                           const int *all_indice, const size_t segment_size, T *gradient,
                                           int *indices) {
  int offset = 0;
  int index = 0;
  size_t segment_data_size = segment_size * sizeof(T);
  for (auto &pair : indice_to_grads) {
    if (distinct_ids.count(pair.first) == 0) {
      continue;
    }
    indices[index++] = pair.first;
    auto ret = memcpy_s(gradient + offset, segment_data_size, pair.second, segment_data_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
    }
    offset += segment_size;
  }
}

template <typename T>
void WorkerProxy<T>::BuildSparseValue(const ::ps::SArray<int> &lengths, const size_t grad_index,
                                      const size_t indice_index, const T *original_data, const T *grads, int *indices,
                                      ::ps::SArray<T> *reduced_data) {
  int offset = 0;
  for (size_t i = 0; i < lengths.size(); i++) {
    if (i != grad_index && i != indice_index) {
      int data_size = lengths[i] * sizeof(T);
      auto ret = memcpy_s(reduced_data->data() + offset, data_size, original_data + offset, data_size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      }
    }
    offset += lengths[i];
  }

  // Fill the reduced gradient
  int grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  int data_size = lengths[grad_index] * sizeof(T);
  auto ret = memcpy_s(reduced_data->data() + grad_offset, data_size, grads, data_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }

  // Fill the reduced indice
  int indice_offset = grad_offset + lengths[grad_index];
  data_size = lengths[indice_index] * sizeof(T);
  T *indice_data = reduced_data->data() + indice_offset;
  T *convert = new T[lengths[indice_index]];
  for (int i = 0; i < lengths[indice_index]; i++) {
    convert[i] = static_cast<T>(indices[i]);
  }
  ret = memcpy_s(indice_data, data_size, convert, data_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  delete[] convert;
}

template <typename T>
void WorkerProxy<T>::BroadcastSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                     std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                     const std::map<int, int> &attr) {
  sliced->resize(server_num_);
  for (int i = 0; i < server_num_; i++) {
    sliced->at(i).first = true;
    sliced->at(i).second = send;
    expected_result_count_[timestamp] += 1;
  }
}

template <typename T>
void WorkerProxy<T>::RoundRobinSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                      const std::map<int, int> &attr) {
  sliced->resize(server_num_);
  auto keys = send.keys;
  auto vals = send.vals;
  auto lens = send.lens;

  int server_id, len;
  ::ps::Key param_key;
  for (size_t i = 0; i < keys.size(); i++) {
    param_key = keys[i];
    server_id = key_to_server_id_[param_key];
    if (!sliced->at(server_id).first) {
      sliced->at(server_id).first = true;
      expected_result_count_[timestamp] += 1;
    }

    ::ps::KVPairs<T> &server_kv_pairs = sliced->at(server_id).second;
    server_kv_pairs.keys.push_back(param_key);
    if (vals.empty()) {
      continue;
    }

    len = lens[i];
    int offset = std::accumulate(lens.begin(), lens.begin() + i, 0);
    auto val_begin = vals.begin() + offset;
    auto val_end = val_begin + len;

    for (auto iter = val_begin; iter != val_end; iter++) {
      server_kv_pairs.vals.push_back(*iter);
    }
    server_kv_pairs.lens.push_back(len);
  }
}

template <typename T>
void WorkerProxy<T>::WorkerInitEmbeddingSlicer(int timestamp, const ::ps::KVPairs<T> &send,
                                               const std::vector<::ps::Range> &,
                                               std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced,
                                               const std::map<int, int> &attrs) {
  sliced->resize(server_num_);
  auto keys = send.keys;
  auto vals = send.vals;
  auto lens = send.lens;

  size_t col_cnt = lens[0] / embedding_row_cnt_[keys[0]];
  const std::vector<::ps::Range> &ranges = *(embedding_table_ranges_[keys[0]]);
  for (size_t i = 0; i < ranges.size(); i++) {
    size_t offset_begin = ranges[i].begin() * col_cnt;
    size_t offset_end = (ranges[i].end() + 1) * col_cnt;
    ::ps::KVPairs<T> kvs;
    kvs.keys = keys;
    kvs.vals = vals.segment(offset_begin, offset_end);
    kvs.lens.push_back(offset_end - offset_begin);
    sliced->at(i).first = true;
    sliced->at(i).second = kvs;
  }
}

template <typename T>
void WorkerProxy<T>::ProcessLookupResult(const ::ps::Message &msg) {
  int ts = msg.meta.timestamp;
  if (msg.meta.pull) {
    CHECK_GE(msg.data.size(), (size_t)2);
    ::ps::KVPairs<T> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
    }
    mutex_.lock();
    lookup_results_[ts].push_back(kvs);
    mutex_.unlock();
  }
  if (lookup_customer_->NumResponse(ts) + 1 == server_num_) {
    const auto &cb = lookup_callbacks_[ts];
    cb();
    lookup_callbacks_.erase(ts);
  }
}

template <typename T>
void WorkerProxy<T>::ProcessResponse(const ::ps::Message &msg) {
  int ts = msg.meta.timestamp;

  if (msg.meta.pull) {
    CHECK_GE(msg.data.size(), (size_t)2);
    ::ps::KVPairs<T> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
    }
    mutex_.lock();
    int rsp_server_rank = ::ps::Postoffice::Get()->IDtoRank(msg.meta.sender);
    gathered_response_[ts][rsp_server_rank] = kvs;
    mutex_.unlock();
    if (general_customer_->NumResponse(ts) + 1 == server_num_) {
      const auto &cb = general_callbacks_[ts];
      cb();
      general_callbacks_.erase(ts);
    }
  }
}

template <typename T>
void WorkerProxy<T>::Send(::ps::Customer *customer, int timestamp, bool push, bool pull, int cmd,
                          const ::ps::KVPairs<T> &kvs, const Slicer &slicer, std::map<int, int> attrs) {
  SlicedKVs sliced;
  slicer(timestamp, kvs, ::ps::Postoffice::Get()->GetServerKeyRanges(), &sliced, attrs);

  for (size_t i = 0; i < sliced.size(); i++) {
    const auto &s = sliced[i];
    if (!s.first) continue;
    ::ps::Message msg;
    msg.meta.app_id = customer->app_id();
    msg.meta.customer_id = customer->customer_id();
    msg.meta.request = true;
    msg.meta.push = push;
    msg.meta.pull = pull;
    msg.meta.head = cmd;
    msg.meta.timestamp = timestamp;
    msg.meta.recver = ::ps::Postoffice::Get()->ServerRankToID(i);
    msg.meta.priority = kvs.priority;
    const auto &kvs = s.second;
    if (kvs.keys.size()) {
      msg.AddData(kvs.keys);
      msg.AddData(kvs.vals);
      if (kvs.lens.size()) {
        msg.AddData(kvs.lens);
      }
    }
    ::ps::Postoffice::Get()->van()->Send(msg);
  }
}
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_WORKER_PROXY_H_
