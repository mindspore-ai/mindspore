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

#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include <unordered_set>
#include "ps/ps.h"
#include "frontend/parallel/ps/util.h"

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
                                    SlicedKVs *sliced)>;
  using ::ps::SimpleApp::obj_;
  explicit WorkerProxy(int app_id, int customer_id, int lookup_customer_id, int general_customer_id)
      : Worker(app_id, customer_id) {
    server_num_ = ::ps::NumServers();
    Util::SetRankId(::ps::MyRank());
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    lookup_customer_ = std::unique_ptr<::ps::Customer>(
      new ::ps::Customer(app_id, lookup_customer_id, std::bind(&WorkerProxy<T>::ProcessLookupResult, this, _1)));
    general_customer_ = std::unique_ptr<::ps::Customer>(
      new ::ps::Customer(app_id, general_customer_id, std::bind(&WorkerProxy<T>::ProcessResponse, this, _1)));
    lookup_slicer_ = std::bind(&WorkerProxy<T>::LookupIdSlicer, this, _1, _2, _3, _4);
    broadcast_slicer_ = std::bind(&WorkerProxy<T>::BroadcastSlicer, this, _1, _2, _3, _4);
    round_robin_slicer_ = std::bind(&WorkerProxy<T>::RoundRobinSlicer, this, _1, _2, _3, _4);
    worker_init_embedding_slicer_ = std::bind(&WorkerProxy<T>::WorkerInitEmbeddingSlicer, this, _1, _2, _3, _4);
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
                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void BroadcastSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                       std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void RoundRobinSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                        std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void WorkerInitEmbeddingSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                 std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void ProcessLookupResult(const ::ps::Message &msg);
  void ProcessResponse(const ::ps::Message &msg);
  void Send(::ps::Customer *customer, int timestamp, bool push, bool pull, int cmd, const ::ps::KVPairs<T> &kvs,
            const Slicer &slicer);
  void AddKeyByHashMod(const ::ps::Key &key);

  int server_num_;
  std::unique_ptr<::ps::Customer> lookup_customer_;
  std::unique_ptr<::ps::Customer> general_customer_;
  std::unordered_map<::ps::Key, std::shared_ptr<std::vector<::ps::Range>>> embedding_table_ranges_;
  std::unordered_map<int, std::vector<::ps::KVPairs<T>>> lookup_results_;
  std::unordered_map<int, ::ps::KVPairs<T>> gathered_response_;
  std::mutex mutex_;
  Slicer lookup_slicer_;
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
    auto &kvs = gathered_response_[ts];
    mutex_.unlock();

    *vals = kvs.vals;
    if (lens) {
      *lens = kvs.lens;
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
                                    std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced) {
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
void WorkerProxy<T>::BroadcastSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                     std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced) {
  sliced->resize(server_num_);
  for (int i = 0; i < server_num_; i++) {
    sliced->at(i).first = true;
    sliced->at(i).second = send;
    expected_result_count_[timestamp] += 1;
  }
}

template <typename T>
void WorkerProxy<T>::RoundRobinSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced) {
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
                                               std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced) {
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
  if (lookup_customer_->NumResponse(ts) == expected_result_count_[ts] - 1) {
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
    for (auto key : kvs.keys) {
      gathered_response_[ts].keys.push_back(key);
    }
    for (auto val : kvs.vals) {
      gathered_response_[ts].vals.push_back(val);
    }
    for (auto len : kvs.lens) {
      gathered_response_[ts].lens.push_back(len);
    }
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
                          const ::ps::KVPairs<T> &kvs, const Slicer &slicer) {
  SlicedKVs sliced;
  slicer(timestamp, kvs, ::ps::Postoffice::Get()->GetServerKeyRanges(), &sliced);

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
