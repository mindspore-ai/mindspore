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
constexpr size_t kInferTimeout = 1800;  // 60*30 30min, next pr will change the solution of endless.

class HealthPointMgr {
 public:
  ~HealthPointMgr() = default;
  HealthPointMgr(const HealthPointMgr &) = delete;
  HealthPointMgr &operator=(const HealthPointMgr &) = delete;
  static HealthPointMgr &GetInstance();

  void DropPoint() {
    std::unique_lock<std::mutex> lock(lock_);
    auto time = std::chrono::microseconds(1);
    auto cond = condition_var_.wait_for(lock, time, [this] { return point_ > 1; });
    if (cond) {
      --point_;
    } else {
      MS_LOG(EXCEPTION) << "Enter endless loop. Please check the code. ";
    }
  }

  void AddPoint() {
    {
      std::lock_guard<std::mutex> lock(lock_);
      ++point_;
    }
    condition_var_.notify_all();
  }

 private:
  HealthPointMgr() = default;
  int point_{1};
  std::mutex lock_;
  std::condition_variable condition_var_;
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

class AsyncEvalResult;
using AsyncEvalResultPtr = std::shared_ptr<AsyncEvalResult>;

using EvaluatorCacheMap =
  std::unordered_map<AbstractBasePtrList, EvalResultPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
using EvalResultCache = NormalCache<AbstractBasePtrList, EvalResultPtr, EvaluatorCacheMap>;

class AsyncEvalResult {
 public:
  AsyncEvalResult() = default;
  ~AsyncEvalResult() = default;
  // Wait
  EvalResultPtr GetResult();
  // Not wait
  EvalResultPtr TryGetResult(int ms = 0);
  void JoinResult(const EvalResultPtr &result);
  std::string ToString();

 private:
  EvalResultPtr result_{nullptr};
  std::mutex lock_;
  std::condition_variable condition_var_;
};

template <typename Type>
class AsyncResult {
 public:
  AsyncResult() = default;
  ~AsyncResult() = default;
  // Wait
  Type GetResult() {
    std::unique_lock<std::mutex> lock(lock_);
    if (result_ != nullptr) {
      return result_;
    }
    auto time = std::chrono::seconds(kInferTimeout);
    // Check if enter endless loop
    HealthPointScopedDrop health_point_check;
    auto cond = condition_var_.wait_for(lock, time, [this] { return result_ != nullptr; });
    if (cond) {
      return result_;
    } else {
      MS_LOG(ERROR) << "Timeout!";
      return nullptr;
    }
  }
  // Not wait
  Type TryGetResult(int ms = 0) {
    std::unique_lock<std::mutex> lock(lock_);
    if (ms == 0) {
      return result_;
    }
    // Check if enter endless loop
    HealthPointScopedDrop health_point_check;
    auto time = std::chrono::microseconds(ms);
    // Wait for ms.
    (void)condition_var_.wait_for(lock, time, [this] { return result_ != nullptr; });
    return result_;
  }
  void JoinResult(const Type &result) {
    MS_EXCEPTION_IF_NULL(result);
    {
      std::lock_guard<std::mutex> lock(lock_);
      result_ = result;
    }
    condition_var_.notify_all();
  }
  std::string ToString() {
    std::ostringstream buffer;
    std::lock_guard<std::mutex> lock(lock_);
    buffer << (result_ == nullptr ? "NOT SET" : result_->ToString());
    return buffer.str();
  }

 private:
  Type result_{nullptr};
  std::mutex lock_;
  std::condition_variable condition_var_;
};

using AsyncAbstractResult = AsyncResult<AbstractBasePtr>;
using AsyncAbstractResultPtr = std::shared_ptr<AsyncAbstractResult>;

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
  static AnalysisResultCacheMgr &GetInstance();
  void Clear();

  using AnalysisConfigAsyncResultMap =
    std::unordered_map<AnfNodeConfigPtr, AsyncEvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigAsyncResultCache =
    MultiThreadCache<AnfNodeConfigPtr, AsyncEvalResultPtr, AnalysisConfigAsyncResultMap>;

  using AnalysisConfigResultMap =
    std::unordered_map<AnfNodeConfigPtr, EvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigResultCache = NormalCache<AnfNodeConfigPtr, EvalResultPtr, AnalysisConfigResultMap>;

  inline void SetValue(const AnfNodeConfigPtr &conf, const EvalResultPtr &arg) { cache_.set(conf, arg); }
  inline EvalResultPtr GetValue(const AnfNodeConfigPtr &conf) { return cache_.get(conf); }

  // Dump all the conf and result
  void DumpCache(const std::string &filename);
  // Wait for async Eval(conf) to finish.
  void Wait();
  void PushTowait(std::future<void> &&future0, std::future<void> &&future1);
  void PushTodo(const AnfNodeConfigPtr &conf);
  void Todo();
  static void UpdateCaller(const std::string &caller);
  static std::string &GetThreadid();

  void InitSwitchValue(const AnfNodeConfigPtr &conf);
  EvalResultPtr GetSwitchValue(const AnfNodeConfigPtr &conf);
  void SetSwitchValue(const AnfNodeConfigPtr &conf, const EvalResultPtr vale);

 private:
  AnalysisResultCacheMgr() = default;
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
