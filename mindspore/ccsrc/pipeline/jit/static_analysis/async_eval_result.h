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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_ASYNC_EVAL_RESULT_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_ASYNC_EVAL_RESULT_H_

#include <iostream>
#include <utility>
#include <future>
#include <thread>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <list>
#include <set>
#include <fstream>
#include <chrono>

#include "pipeline/jit/static_analysis/static_analysis.h"

namespace mindspore {
namespace abstract {

class AsyncInferTask;
class AsyncAbstract;
using AsyncInferTaskPtr = std::shared_ptr<AsyncInferTask>;
using AsyncAbstractPtr = std::shared_ptr<AsyncAbstract>;
class AnalysisSchedule {
 public:
  ~AnalysisSchedule() { Stop(); }
  AnalysisSchedule(const AnalysisSchedule &) = delete;
  AnalysisSchedule &operator=(const AnalysisSchedule &) = delete;
  static AnalysisSchedule &GetInstance() { return instance_; }
  static void SetThreadID(const std::string &caller);
  static std::string &GetThreadID();
  void HandleException(const std::exception &ex);
  void Stop() {
    notExit_ = false;
    MS_LOG(DEBUG) << " Set AnalysisSchedule::Exit . The active thread count: " << activate_threads_.size()
                  << " The infer_thread_count: " << infer_thread_count_
                  << " schedule list size: " << scheduleList_.size();
  }
  void Wait();
  void Add2Schedule(const AsyncInferTaskPtr &async_infer_task_ptr);
  void Yield(const AsyncInferTask *asyncTask);

  void EnterWaiting() {
    {
      MS_LOG(DEBUG) << " Require activate_thread_lock. The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << scheduleList_.size();
      std::lock_guard<std::mutex> activeLock(activate_thread_lock_);
      activate_threads_.clear();
      MS_LOG(DEBUG) << " Get activate_thread_lock. The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << scheduleList_.size() << " thread: " << GetThreadID() + " "
                    << (activate_threads_.size() > 0 ? activate_threads_.begin()->c_str() : "");
    }
    activate_thread_cv_.notify_one();
  }

  void IncreaseThreadCount() {
    infer_thread_count_.fetch_add(1);
    MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size()
                  << " The infer_thread_count: " << infer_thread_count_
                  << " schedule list size: " << scheduleList_.size();
  }

  void DecreaseThreadCount() {
    {
      std::lock_guard<std::mutex> threadNumLock(infer_thread_lock_);
      infer_thread_count_.fetch_sub(1);
    }
    infer_thread_cv_.notify_one();

    {
      std::lock_guard<std::mutex> activeLock(activate_thread_lock_);
      activate_threads_.clear();
      MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << scheduleList_.size() << " thread: " << GetThreadID() + " "
                    << (activate_threads_.size() > 0 ? activate_threads_.begin()->c_str() : "");
    }
    activate_thread_cv_.notify_one();
  }

 private:
  void Schedule();
  bool SetNextReady();
  void Start() {
    auto thread = std::thread([this] { Schedule(); });
    thread.detach();
  }
  AnalysisSchedule() { Start(); }
  static AnalysisSchedule instance_;
  std::atomic<int> infer_thread_count_{0};
  bool notExit_{true};
  std::mutex infer_thread_lock_;
  std::condition_variable infer_thread_cv_;
  std::mutex activate_thread_lock_;
  std::condition_variable activate_thread_cv_;
  std::list<AsyncInferTaskPtr> scheduleList_;
  std::set<std::string> activate_threads_;
};

template <typename KeyType, typename ValueType, typename CacheType>
class MultiThreadCache {
 public:
  using iterator = typename CacheType::iterator;
  using const_iterator = typename CacheType::const_iterator;

  ValueType get(const KeyType &key) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void set(const KeyType &key, const ValueType &data) {
    std::lock_guard<std::mutex> lock(lock_);
    cache_[key] = data;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(lock_);
    cache_.clear();
  }

  size_t size() { return cache_.size(); }

  bool empty() { return size() == 0; }

  std::string dump() {
    std::ostringstream buf;
    for (auto &item : cache_) {
      buf << "{" << item.first->ToString() << ": " << item.second->ToString() << "}" << std::endl;
    }
    return buf.str();
  }

  iterator begin() { return cache_.begin(); }
  iterator end() { return cache_.end(); }

  const_iterator begin() const { return cache_.cbegin(); }
  const_iterator end() const { return cache_.cend(); }

  const_iterator cbegin() const { return cache_.cbegin(); }
  const_iterator cend() const { return cache_.cend(); }

 private:
  std::mutex lock_;
  CacheType cache_;
};

template <typename KeyType, typename ValueType, typename CacheType>
class NormalCache {
 public:
  using iterator = typename CacheType::iterator;
  using const_iterator = typename CacheType::const_iterator;

  ValueType get(const KeyType &key) const {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void set(const KeyType &key, const ValueType &data) { cache_[key] = data; }

  void clear() { cache_.clear(); }

  size_t size() const { return cache_.size(); }

  bool empty() const { return size() == 0; }

  std::string dump() const {
    std::ostringstream buf;
    for (auto &item : cache_) {
      buf << "{" << item.first->ToString() << ": " << item.second->ToString() << "}" << std::endl;
    }
    return buf.str();
  }

  iterator begin() { return cache_.begin(); }
  iterator end() { return cache_.end(); }

  const_iterator begin() const { return cache_.cbegin(); }
  const_iterator end() const { return cache_.cend(); }

  const_iterator cbegin() const { return cache_.cbegin(); }
  const_iterator cend() const { return cache_.cend(); }

 private:
  CacheType cache_;
};

class AsyncAbstract {
 public:
  AsyncAbstract() = default;
  ~AsyncAbstract() = default;
  AbstractBasePtr TryGetResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_;
  }
  bool HasResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_ != nullptr;
  }
  void SetResult(const AbstractBasePtr &result) {
    MS_EXCEPTION_IF_NULL(result);
    std::lock_guard<std::mutex> lock(lock_);
    result_ = result;
  }

  std::string ToString() {
    std::ostringstream buffer;
    std::lock_guard<std::mutex> lock(lock_);
    buffer << (result_ == nullptr ? "NOT SET" : result_->ToString());
    return buffer.str();
  }

 private:
  std::mutex lock_;
  AbstractBasePtr result_{nullptr};
};

class AsyncInferTask {
 public:
  explicit AsyncInferTask(const std::string &threadId, const AsyncAbstractPtr &abstract)
      : threadId_(threadId), abstract_ptr_(abstract) {}
  ~AsyncInferTask() = default;

  static AsyncInferTaskPtr MakeShared(const AsyncAbstractPtr &abstract, const std::string &threadId = "") {
    std::string thread_id = threadId;
    if (thread_id == "") {
      thread_id = AnalysisSchedule::GetInstance().GetThreadID();
    }
    MS_EXCEPTION_IF_NULL(abstract);
    auto ret = std::make_shared<AsyncInferTask>(thread_id, abstract);
    MS_EXCEPTION_IF_NULL(ret);
    return ret;
  }

  bool HasResult() { return abstract_ptr_->HasResult(); }
  int Ready() const { return ready_; }
  std::string ThreadID() const { return threadId_; }

  AbstractBasePtr GetResult() {
    std::unique_lock<std::mutex> lock(lock_);
    if (ready_) {
      ProcessResult();
      return abstract_ptr_->TryGetResult();
    }
    // Avoid to dead lock between AsyncAbstract::lock and AnalysisSchedule::activate_thread_lock_
    lock.unlock();
    AnalysisSchedule::GetInstance().Yield(this);

    lock.lock();
    MS_LOG(DEBUG) << this << " after enter waiting ready: " << ready_ << " thread id:" << threadId_
                  << " GetThreadId: " << AnalysisSchedule::GetInstance().GetThreadID();
    condition_var_.wait(lock, [this] { return ready_; });
    MS_LOG(DEBUG) << this << " received notify and wake up: " << ready_ << " thread id:" << threadId_
                  << " GetThreadId: " << AnalysisSchedule::GetInstance().GetThreadID();
    ProcessResult();
    auto ans = abstract_ptr_->TryGetResult();
    MS_EXCEPTION_IF_NULL(ans);
    return ans;
  }

  void SetReady() {
    MS_LOG(DEBUG) << this << " want to set ready.";
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 1;  // Set the first bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_ << " result: " << abstract_ptr_->TryGetResult().get()
                    << " threadId: " << threadId_;
    }
    condition_var_.notify_one();
  }

  void SetException() {
    MS_LOG(DEBUG) << this << " want to set ready.";
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 2;  // Set the second bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_;
    }
    condition_var_.notify_one();
  }

  void SetEndLessLoopException() {
    MS_LOG(DEBUG) << this << " want to set ready.";
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 4;  // Set the third bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_;
    }
    condition_var_.notify_one();
  }

 private:
  void ClearReady() {
    ready_ = ready_ & 6;  // Set first bit = 0
    MS_LOG(DEBUG) << this << " ready: " << ready_ << " result: " << abstract_ptr_->TryGetResult().get();
  }
  void HandleEndLessLoopException() {
    // Get third bit
    if (ready_ & 4) {
      ready_ = ready_ & 3;  // Set the third bit = 0 , Only trigger once.
      MS_LOG(EXCEPTION) << "There isn't any branch that can be evaluated. \n"
                        << "Please check the code if it's has the infinite recursion or loop.\n"
                        << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
    }
  }
  void ProcessResult() {
    ClearReady();  // Clear nomal ready flag
    HandleEndLessLoopException();
    StaticAnalysisException::Instance().CheckException();
    MS_LOG(DEBUG) << this << " Success to GetResult. ready: " << ready_ << " threadId: " << threadId_
                  << " GetThreadId:" << AnalysisSchedule::GetInstance().GetThreadID()
                  << " result: " << abstract_ptr_->TryGetResult().get();
  }
  std::string threadId_;
  AsyncAbstractPtr abstract_ptr_;
  std::mutex lock_;
  std::condition_variable condition_var_;
  size_t ready_{0};  // 0: not ready, bit 1 = 1: ready, bit 2 = 1: exception, bit 3 = 1: endless loop
};

using EvaluatorCacheMap =
  std::unordered_map<AbstractBasePtrList, EvalResultPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
using EvalResultCache = NormalCache<AbstractBasePtrList, EvalResultPtr, EvaluatorCacheMap>;

class EvaluatorCacheMgr {
 public:
  EvaluatorCacheMgr() = default;
  ~EvaluatorCacheMgr() = default;

  void Clear() { eval_result_cache_.clear(); }
  const EvalResultCache &GetCache() { return eval_result_cache_; }
  EvalResultPtr GetValue(const AbstractBasePtrList &key) { return eval_result_cache_.get(key); }
  void SetValue(const AbstractBasePtrList &key, const EvalResultPtr &arg) { eval_result_cache_.set(key, arg); }
  size_t GetSize() { return eval_result_cache_.size(); }

 private:
  EvalResultCache eval_result_cache_;
};

// AnalysisCache
class AnalysisResultCacheMgr {
 public:
  using AnalysisConfigResultMap =
    std::unordered_map<AnfNodeConfigPtr, EvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigResultCache = NormalCache<AnfNodeConfigPtr, EvalResultPtr, AnalysisConfigResultMap>;
  using const_iterator = typename AnalysisConfigResultCache::const_iterator;

  ~AnalysisResultCacheMgr() = default;
  AnalysisResultCacheMgr(const AnalysisResultCacheMgr &) = delete;
  AnalysisResultCacheMgr &operator=(const AnalysisResultCacheMgr &) = delete;
  static AnalysisResultCacheMgr &GetInstance() { return instance_; }
  void Clear();
  inline void SetValue(const AnfNodeConfigPtr &conf, const EvalResultPtr &arg) { cache_.set(conf, arg); }
  inline EvalResultPtr GetValue(const AnfNodeConfigPtr &conf) { return cache_.get(conf); }
  void PushTodo(const AnfNodeConfigPtr &conf);
  void Todo();
  void InitSwitchValue(const AnfNodeConfigPtr &conf);
  AbstractBasePtr GetSwitchValue(const AnfNodeConfigPtr &conf);
  AbstractBasePtr TryGetSwitchValue(const AnfNodeConfigPtr &conf);
  void SetSwitchValue(const AnfNodeConfigPtr &conf, const AbstractBasePtr &vale);
  const_iterator begin() { return cache_.begin(); }
  const_iterator end() { return cache_.end(); }

 private:
  using AnalysisConfigAsyncResultMap =
    std::unordered_map<AnfNodeConfigPtr, AsyncAbstractPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigAsyncResultCache =
    MultiThreadCache<AnfNodeConfigPtr, AsyncAbstractPtr, AnalysisConfigAsyncResultMap>;
  AnalysisResultCacheMgr() = default;
  static AnalysisResultCacheMgr instance_;
  std::mutex lock_;
  std::mutex todo_lock_;
  std::list<AnfNodeConfigPtr> todo_;
  AnalysisConfigResultCache cache_;
  AnalysisConfigAsyncResultCache switch_cache_;
};

std::string ArgsToString(const AbstractBasePtrList &args_spec_list);

inline std::string GetInferThread() { return std::string(" INFER:") + AnalysisSchedule::GetThreadID() + ":"; }

}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_ASYNC_EVAL_RESULT_H_
