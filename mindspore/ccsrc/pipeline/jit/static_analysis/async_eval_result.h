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
#include <fstream>

#include "pipeline/jit/static_analysis/static_analysis.h"

namespace mindspore {
namespace abstract {

class AsyncAbstract;
using AsyncAbstractPtr = std::shared_ptr<AsyncAbstract>;
class AnalysisSchedule {
 public:
  ~AnalysisSchedule() = default;
  AnalysisSchedule(const AnalysisSchedule &) = delete;
  AnalysisSchedule &operator=(const AnalysisSchedule &) = delete;
  static AnalysisSchedule &GetInstance() { return instance_; }
  static void SetThreadID(const std::string &caller);
  static std::string &GetThreadID();
  void HandleException(const std::exception &ex);
  std::string GetExtendException() { return exceptionStream_.str(); }
  void Wait();

  void Reset() {
    activeThreadCount_ = 1;
    threadNum_ = 0;
    exceptionStream_.clear();
  }

  void SetNextRunnable() {
    std::lock_guard<std::mutex> lock(lock_);
    SetNextRunnableImpl();
  }

  void Check() {
    MS_LOG(DEBUG) << "The active thread count: " << activeThreadCount_;
    if (activeThreadCount_ == 0) {
      SetNextRunnableImpl();
    } else if (activeThreadCount_ < 0) {
      MS_LOG(ERROR) << "There is something wrong. active thread count: " << activeThreadCount_;
    }
  }

  void EnterWaiting() {
    std::lock_guard<std::mutex> lock(lock_);
    --activeThreadCount_;
    MS_LOG(DEBUG) << this << " The active thread count: " << activeThreadCount_;
    Check();
  }

  void LeaveWaiting() {
    std::lock_guard<std::mutex> lock(lock_);
    ++activeThreadCount_;
    MS_LOG(DEBUG) << this << " The active thread count: " << activeThreadCount_;
  }

  void Add2Schedule(const AsyncAbstractPtr &asyncAbastract) {
    std::lock_guard<std::mutex> lock(lock_);
    asyncAbstractList_.push_back(asyncAbastract);
  }
  void IncreaseThreadCount() {
    std::lock_guard<std::mutex> lock(lock_);
    ++threadNum_;
    ++activeThreadCount_;
    MS_LOG(DEBUG) << "The active thread count: " << activeThreadCount_;
  }
  void DecreaseThreadCount() {
    {
      std::lock_guard<std::mutex> threadNumLock(lock_);
      --threadNum_;
    }
    condition_var_.notify_one();

    std::lock_guard<std::mutex> activeLock(lock_);
    --activeThreadCount_;
    MS_LOG(DEBUG) << "The active thread count: " << activeThreadCount_;
    Check();
  }

 private:
  void SetNextRunnableImpl();
  AnalysisSchedule() = default;
  static AnalysisSchedule instance_;
  int activeThreadCount_{1};
  int threadNum_{0};
  std::mutex lock_;
  std::condition_variable condition_var_;
  std::list<AsyncAbstractPtr> asyncAbstractList_;
  std::ostringstream exceptionStream_;
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

class AsyncAbstract : public std::enable_shared_from_this<AsyncAbstract> {
 public:
  AsyncAbstract() = default;
  ~AsyncAbstract() = default;
  // Wait
  AbstractBasePtr GetResult() {
    StaticAnalysisException::Instance().CheckException();
    while (true) {
      ++count_;
      // The active thread count should be dropped if it can't run. It will be added when it can run.
      MS_LOG(DEBUG) << this << " continue runnable: " << runnable_ << " result: " << (result_ ? result_.get() : 0);
      bool hasEnterWaiting = false;
      if (!runnable_) {
        AnalysisSchedule::GetInstance().EnterWaiting();
        hasEnterWaiting = true;
      }
      MS_LOG(DEBUG) << this << " runnable: " << runnable_ << " result: " << (result_ ? result_.get() : 0);
      {
        std::unique_lock<std::mutex> lock(lock_);
        condition_var_.wait(lock, [this] { return runnable_; });
      }
      if (hasEnterWaiting) {
        AnalysisSchedule::GetInstance().LeaveWaiting();
      }
      MS_LOG(DEBUG) << this << " continue runnable: " << runnable_ << " result: " << (result_ ? result_.get() : 0);

      StaticAnalysisException::Instance().CheckException();
      SetUnrunnable();
      if (result_ != nullptr) {
        MS_LOG(DEBUG) << this << " Return  result: " << (result_ ? result_.get() : 0);
        return result_;
      }
      // Push to list
      AnalysisSchedule::GetInstance().Add2Schedule(shared_from_this());
      // Notify the next asyncAbastract to run.
      AnalysisSchedule::GetInstance().SetNextRunnable();
      MS_LOG(DEBUG) << this << " SetNextRunnable "
                    << " runnable: " << runnable_ << " result: " << (result_ ? result_.get() : 0);
    }
  }

  void SetRunnable() {
    MS_LOG(DEBUG) << this << " Runnable.";
    {
      std::lock_guard<std::mutex> lock(lock_);
      runnable_ = true;
    }
    condition_var_.notify_one();
  }
  void SetUnrunnable() {
    std::lock_guard<std::mutex> lock(lock_);
    runnable_ = false;
  }

  int count() const { return count_; }

  bool HasResult() { return result_ != nullptr; }
  // Not wait
  AbstractBasePtr TryGetResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_;
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
  std::condition_variable condition_var_;
  bool runnable_{false};
  int count_{0};
  AbstractBasePtr result_{nullptr};
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
