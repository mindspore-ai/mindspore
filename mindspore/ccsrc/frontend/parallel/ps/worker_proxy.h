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
  explicit WorkerProxy(int app_id, int customer_id, int lookup_customer_id) : Worker(app_id, customer_id) {
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    lookup_customer_ = std::unique_ptr<::ps::Customer>(
      new ::ps::Customer(app_id, lookup_customer_id, std::bind(&WorkerProxy<T>::ProcessLookupResult, this, _1)));
    lookup_slicer_ = std::bind(&WorkerProxy<T>::LookupIdSlicer, this, _1, _2, _3, _4);
    broadcast_slicer_ = std::bind(&WorkerProxy<T>::BroadcastSlicer, this, _1, _2, _3, _4);
  }
  ~WorkerProxy() override = default;

  void AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count);
  void EmbeddingLookup(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids,
                       const ::ps::SArray<int> &lens, ::ps::SArray<T> *outs, int cmd = 0, const Callback &cb = nullptr,
                       int priority = 0);
  int InitEmbeddingTable(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                         const ::ps::SArray<int> &lens = {}, const Callback &cb = nullptr, int priority = 0);
  bool IsReadyForPush(const Key &key);
  bool IsReadyForPull(const Key &key);
  void PushData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals, const ::ps::SArray<int> &lens = {},
                int cmd = 0, int priority = 0);
  void Finalize();

 private:
  template <typename C>
  int AddLookupCB(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<int> &lookup_ids, C *vals, int cmd,
                  const Callback &cb);
  void LookupIdSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                      std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void BroadcastSlicer(int timestamp, const ::ps::KVPairs<T> &send, const std::vector<::ps::Range> &,
                       std::vector<std::pair<bool, ::ps::KVPairs<T>>> *sliced);
  void ProcessLookupResult(const ::ps::Message &msg);
  void Send(::ps::Customer *customer, int timestamp, bool push, bool pull, int cmd, const ::ps::KVPairs<T> &kvs,
            const Slicer &slicer);

  std::unique_ptr<::ps::Customer> lookup_customer_;
  std::unordered_map<::ps::Key, std::shared_ptr<std::vector<::ps::Range>>> embedding_table_ranges_;
  std::unordered_map<int, std::vector<::ps::KVPairs<T>>> lookup_results_;
  std::mutex mutex_;
  Slicer lookup_slicer_;
  Slicer broadcast_slicer_;
  std::unordered_map<int, Callback> lookup_callbacks_;
  std::unordered_map<int, int> expected_result_count_;
};

template <typename T>
void WorkerProxy<T>::AddEmbeddingTable(const ::ps::Key &key, const size_t &row_count) {
  uint64_t begin = 0;
  uint64_t end = 0;
  int server_num = ::ps::NumServers();
  for (int i = 0; i < server_num; i++) {
    int local_row_cnt = Util::LocalShard(row_count, i, server_num);
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
  int server_num = ::ps::NumServers();
  int expect_rt_count = expected_result_count_[ts];
  lookup_customer_->AddResponse(ts, server_num - expect_rt_count);
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
  this->Wait(this->ZPull({key}, &result, nullptr, kCheckReadyForPushCmd));
  if (result[0] > 0) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
bool WorkerProxy<T>::IsReadyForPull(const Key &key) {
  ::ps::SArray<T> result(1, 0);
  this->Wait(this->ZPull({key}, &result, nullptr, kCheckReadyForPullCmd));
  if (result[0] > 0) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
void WorkerProxy<T>::PushData(const ::ps::SArray<::ps::Key> &keys, const ::ps::SArray<T> &vals,
                              const ::ps::SArray<int> &lens, int cmd, int priority) {
  int ts = obj_->NewRequest(::ps::kServerGroup);
  ::ps::KVPairs<T> kvs;
  kvs.keys = keys;
  kvs.vals = vals;
  kvs.lens = lens;
  kvs.priority = priority;
  Send(obj_, ts, true, false, cmd, kvs, broadcast_slicer_);
  obj_->WaitRequest(ts);
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

    auto &s = kvs[0];
    *lookup_result = s.vals;

    mutex_.lock();
    lookup_results_.erase(ts);
    mutex_.unlock();
    if (cb) cb();
  };
  lookup_callbacks_[ts] = callback;
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
    auto &kvs = sliced->at(i).second;

    kvs.keys.push_back(key);
    kvs.vals.push_back(0.0f);
    for (size_t j = 0; j < id_size; j++) {
      kvs.keys.push_back(lookup_ids[j]);
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
  auto server_num = ::ps::Postoffice::Get()->num_servers();
  sliced->resize(server_num);
  for (int i = 0; i < server_num; i++) {
    sliced->at(i).first = true;
    sliced->at(i).second = send;
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
