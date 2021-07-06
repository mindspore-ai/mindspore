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
class HealthPointMgr {
 public:
  ~HealthPointMgr() = default;
  HealthPointMgr(const HealthPointMgr &) = delete;
  HealthPointMgr &operator=(const HealthPointMgr &) = delete;
  static HealthPointMgr &GetInstance() { return instance_; }
  void Clear();
  void SetNextRunable();
  void HandleException();

  void CheckPoint() {
    MS_LOG(DEBUG) << "The Health Point is " << point_;
    if (point_ == 0) {
      SetNextRunable();
    } else if (point_ < 0) {
      MS_LOG(EXCEPTION) << "There is something wrong.";
    }
  }

  void DropPoint() {
    std::lock_guard<std::recursive_mutex> lock(lock_);
    --point_;
    CheckPoint();
  }

  void AddPoint() {
    std::lock_guard<std::recursive_mutex> lock(lock_);
    ++point_;
  }

  int point() { return point_; }

  void Add2Schedule(const AsyncAbstractPtr &asyncAbastract) {
    std::lock_guard<std::recursive_mutex> lock(lock_);
    asyncAbstractList_.push_back(asyncAbastract);
  }

 private:
  HealthPointMgr() = default;
  static HealthPointMgr instance_;
  int point_{1};
  std::recursive_mutex lock_;
  std::list<AsyncAbstractPtr> asyncAbstractList_;
};

class HealthPointScopedDrop {
 public:
  HealthPointScopedDrop() { HealthPointMgr::GetInstance().DropPoint(); }
  ~HealthPointScopedDrop() { HealthPointMgr::GetInstance().AddPoint(); }
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
      buf << "{" << item.first->ToString() << ":" << item.second->ToString() << "}" << std::endl;
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

  ValueType get(const KeyType &key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void set(const KeyType &key, const ValueType &data) { cache_[key] = data; }

  void clear() { cache_.clear(); }

  size_t size() { return cache_.size(); }

  bool empty() { return size() == 0; }

  std::string dump() {
    std::ostringstream buf;
    for (auto &item : cache_) {
      buf << "{" << item.first->ToString() << ":" << item.second->ToString() << "}" << std::endl;
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
    std::unique_lock<std::mutex> lock(lock_);
    while (true) {
      ++count_;
      // The point should be dropped if it can't run. It will be added when it can run.
      bool hasDropPoint = false;
      if (!runable_) {
        HealthPointMgr::GetInstance().DropPoint();
        hasDropPoint = true;
      }

      MS_LOG(DEBUG) << this << " runable: " << runable_ << " result: " << (result_ ? result_.get() : 0);
      condition_var_.wait(lock, [this] { return runable_; });
      MS_LOG(DEBUG) << this << " continue runable: " << runable_ << " result: " << (result_ ? result_.get() : 0);
      StaticAnalysisException::Instance().CheckException();
      runable_ = false;
      if (result_ != nullptr) {
        if (hasDropPoint) {
          HealthPointMgr::GetInstance().AddPoint();
        }
        MS_LOG(DEBUG) << this << " Return  result: " << (result_ ? result_.get() : 0);
        return result_;
      }
      // Push to list
      HealthPointMgr::GetInstance().Add2Schedule(shared_from_this());
      if (hasDropPoint) {
        HealthPointMgr::GetInstance().AddPoint();
      }
      // Notify the next asyncAbastract to run.
      HealthPointMgr::GetInstance().SetNextRunable();
      MS_LOG(DEBUG) << this << " SetNextRunable "
                    << " runable: " << runable_ << " result: " << (result_ ? result_.get() : 0)
                    << " point:" << HealthPointMgr::GetInstance().point();
    }
    return nullptr;
  }
  void SetRunable() {
    MS_LOG(DEBUG) << this << " Runable.";
    runable_ = true;
    condition_var_.notify_one();
  }
  int count() { return count_; }

  bool HasResult() { return result_ != nullptr; }
  // Not wait
  AbstractBasePtr TryGetResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_;
  }
  void JoinResult(const AbstractBasePtr &result) {
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
  bool runable_{false};
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
  EvalResultCache &GetCache() { return eval_result_cache_; }
  EvalResultPtr GetValue(const AbstractBasePtrList &key) { return eval_result_cache_.get(key); }
  void SetValue(const AbstractBasePtrList &key, const EvalResultPtr &arg) { eval_result_cache_.set(key, arg); }
  size_t GetSize() { return eval_result_cache_.size(); }

 private:
  EvalResultCache eval_result_cache_;
};

// AnalysisCache
class AnalysisResultCacheMgr {
 public:
  ~AnalysisResultCacheMgr() = default;
  AnalysisResultCacheMgr(const AnalysisResultCacheMgr &) = delete;
  AnalysisResultCacheMgr &operator=(const AnalysisResultCacheMgr &) = delete;
  static AnalysisResultCacheMgr &GetInstance() { return instance_; }
  void Clear();
  inline void SetValue(const AnfNodeConfigPtr &conf, const EvalResultPtr &arg) { cache_.set(conf, arg); }
  inline EvalResultPtr GetValue(const AnfNodeConfigPtr &conf) { return cache_.get(conf); }
  // Wait for async Eval(conf) to finish.
  void Wait();
  void PushTowait(std::future<void> &&future0, std::future<void> &&future1);
  void PushTodo(const AnfNodeConfigPtr &conf);
  void Todo();
  static void UpdateCaller(const std::string &caller);
  static std::string &GetThreadid();
  void InitSwitchValue(const AnfNodeConfigPtr &conf);
  AbstractBasePtr GetSwitchValue(const AnfNodeConfigPtr &conf);
  AbstractBasePtr TryGetSwitchValue(const AnfNodeConfigPtr &conf);
  void SetSwitchValue(const AnfNodeConfigPtr &conf, const AbstractBasePtr vale);

 private:
  using AnalysisConfigAsyncResultMap =
    std::unordered_map<AnfNodeConfigPtr, AsyncAbstractPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigAsyncResultCache =
    MultiThreadCache<AnfNodeConfigPtr, AsyncAbstractPtr, AnalysisConfigAsyncResultMap>;

  using AnalysisConfigResultMap =
    std::unordered_map<AnfNodeConfigPtr, EvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigResultCache = NormalCache<AnfNodeConfigPtr, EvalResultPtr, AnalysisConfigResultMap>;

  AnalysisResultCacheMgr() = default;
  static AnalysisResultCacheMgr instance_;
  std::mutex lock_;
  std::list<std::future<void>> waiting_;
  std::mutex todo_lock_;
  std::list<AnfNodeConfigPtr> todo_;
  AnalysisConfigResultCache cache_;
  AnalysisConfigAsyncResultCache switch_cache_;
};

std::string ArgsToString(const AbstractBasePtrList &args_spec_list);

inline std::string GetInferThread() { return std::string(" INFER:") + AnalysisResultCacheMgr::GetThreadid() + ":"; }

}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_ASYNC_EVAL_RESULT_H_
