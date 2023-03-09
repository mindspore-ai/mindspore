/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include <functional>
#include <list>
#include <set>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <mutex>

#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace abstract {

class AsyncInferTask;
class AsyncAbstract;
class AsyncAbstractFuncAtom;
using AsyncInferTaskPtr = std::shared_ptr<AsyncInferTask>;
using AsyncAbstractPtr = std::shared_ptr<AsyncAbstract>;
class AnalysisSchedule {
 public:
  ~AnalysisSchedule() = default;
  AnalysisSchedule(const AnalysisSchedule &) = delete;
  AnalysisSchedule &operator=(const AnalysisSchedule &) = delete;
  static AnalysisSchedule &GetInstance() {
    static AnalysisSchedule instance;
    return instance;
  }
  static void set_thread_id(const std::string &thread_id) { thread_id_ = thread_id; }
  static std::string &thread_id() { return thread_id_; }
  void HandleException(const std::exception &ex);
  void Stop();
  void Wait();
  void Add2Schedule(const AsyncInferTaskPtr &async_infer_task_ptr);
  void WaitForRun() const;
  void YieldTask(AsyncInferTask *asyncTask);

  void EnterWaiting() {
    {
      std::lock_guard<std::mutex> activeLock(activate_thread_lock_);
      (void)activate_threads_.erase(AnalysisSchedule::thread_id());
      MS_LOG(DEBUG) << "Infer return to main thread.";
    }
    activate_thread_cv_.notify_one();
  }

  void IncreaseThreadCount() {
    infer_thread_count_.fetch_add(1);
    MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size()
                  << " The infer_thread_count: " << infer_thread_count_
                  << " schedule list size: " << schedule_list_.size();
  }

  void DecreaseThreadCount() {
    {
      std::lock_guard<std::mutex> threadNumLock(infer_thread_lock_);
      infer_thread_count_.fetch_sub(1);
    }
    infer_thread_cv_.notify_one();

    {
      std::lock_guard<std::mutex> active_lock(activate_thread_lock_);
      (void)activate_threads_.erase(AnalysisSchedule::thread_id());
      MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << schedule_list_.size() << " thread: " << thread_id() + " "
                    << (activate_threads_.size() > 0 ? activate_threads_.begin()->c_str() : "");
    }
    activate_thread_cv_.notify_one();
  }

 private:
  void Schedule();
  void SetNextReady();
  void Start() {
    auto thread = std::thread([this] { Schedule(); });
    thread.detach();
  }
  AnalysisSchedule() { Start(); }
  std::atomic<int> infer_thread_count_{0};
  bool run_{true};
  std::mutex infer_thread_lock_;
  std::condition_variable infer_thread_cv_;
  std::mutex activate_thread_lock_;
  std::condition_variable activate_thread_cv_;
  std::list<AsyncInferTaskPtr> schedule_list_;
  std::set<std::string> activate_threads_;
  const std::string kStateStop = "Stop";
  static thread_local std::string thread_id_;
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

  const_iterator find(const KeyType &key) const { return cache_.find(key); }

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
  explicit AsyncAbstract(std::shared_ptr<AsyncAbstract> switchAbstract = nullptr) : switchAbstract_(switchAbstract) {}
  ~AsyncAbstract() = default;
  AbstractBasePtr GetResult();
  AbstractBasePtr TryGetResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_;
  }
  bool HasResult() {
    std::lock_guard<std::mutex> lock(lock_);
    return result_ != nullptr;
  }
  void set_result(const AbstractBasePtr &result) {
    std::lock_guard<std::mutex> lock(lock_);
    result_ = result;
  }

  void ClearPossibleResult();

  std::string ToString() {
    std::ostringstream buffer;
    std::lock_guard<std::mutex> lock(lock_);
    buffer << (result_ == nullptr ? "NOT SET" : result_->ToString());
    return buffer.str();
  }

  bool SetPossibleResult();

 private:
  std::mutex lock_;
  AbstractBasePtr result_{nullptr};
  bool not_copy_from_other_{true};
  std::shared_ptr<AsyncAbstract> switchAbstract_;
};

// Wrap AsyncAbstract, so it can work with Join method of AbstractFunction.
class AsyncAbstractFuncAtom : public AbstractFuncAtom {
 public:
  AsyncAbstractFuncAtom(const AsyncAbstractPtr &async_abstract, const std::vector<std::size_t> &index)
      : async_abstract_(async_abstract), index_(index) {}
  ~AsyncAbstractFuncAtom() = default;
  MS_DECLARE_PARENT(AsyncAbstractFuncAtom, AbstractFuncAtom);

  static std::shared_ptr<AsyncAbstractFuncAtom> MakeShared(const AsyncAbstractPtr &async_abstract,
                                                           const std::vector<std::size_t> &index) {
    MS_EXCEPTION_IF_NULL(async_abstract);
    auto ret = std::make_shared<AsyncAbstractFuncAtom>(async_abstract, index);
    MS_EXCEPTION_IF_NULL(ret);
    return ret;
  }

  AbstractFunctionPtr Copy() const override { return MakeShared(async_abstract_, index_); }

  bool operator==(const AbstractFunction &other) const override {
    if (!other.isa<AsyncAbstractFuncAtom>()) {
      return false;
    }
    auto other_async = static_cast<const AsyncAbstractFuncAtom *>(&other);
    MS_EXCEPTION_IF_NULL(other_async);
    if (index_ != other_async->index_) {
      return false;
    }
    if (async_abstract_ == other_async->async_abstract_) {
      return true;
    }
    auto abs = async_abstract_->TryGetResult();
    auto other_abs = other_async->async_abstract_->TryGetResult();
    if (abs != nullptr && other_abs != nullptr) {
      return *abs == *other_abs;
    } else {
      return false;
    }
  }

  std::size_t hash() const override {
    std::size_t hash_index = 0;
    for (const auto i : index_) {
      hash_index = hash_combine(hash_index, std::hash<std::size_t>{}(i));
    }
    return hash_index;
  }

  AbstractFunctionPtr GetUnique() override;

  std::string ToString() const override;

 private:
  // Resolved AbstractFunction after fully analyzed.
  AbstractFunctionPtr resolved_{nullptr};
  // Before resolved, use the following two items to track.
  const AsyncAbstractPtr async_abstract_;
  const std::vector<std::size_t> index_;
};
using AsyncAbstractFuncAtomPtr = std::shared_ptr<AsyncAbstractFuncAtom>;

class AsyncInferTask {
 public:
  explicit AsyncInferTask(const std::string &thread_id, const AsyncAbstractPtr &abstract)
      : thread_id_(thread_id), abstract_ptr_(abstract) {
    MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " : " << this;
  }
  ~AsyncInferTask() { MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " : " << this; }

  static AsyncInferTaskPtr MakeShared(const AsyncAbstractPtr &abstract, const std::string &thread = "") {
    std::string thread_id = thread;
    if (thread_id == "") {
      thread_id = AnalysisSchedule::thread_id();
    }
    MS_EXCEPTION_IF_NULL(abstract);
    auto ret = std::make_shared<AsyncInferTask>(thread_id, abstract);
    MS_EXCEPTION_IF_NULL(ret);
    return ret;
  }

  bool HasResult() { return abstract_ptr_->HasResult(); }
  bool SetPossibleResult() { return abstract_ptr_->SetPossibleResult(); }
  int ready() {
    std::lock_guard<std::mutex> lock(lock_);
    return SizeToInt(ready_);
  }
  std::string thread_id() const { return thread_id_; }

  AbstractBasePtr GetResult() {
    StaticAnalysisException::Instance().CheckException();
    AnalysisSchedule::GetInstance().YieldTask(this);
    std::unique_lock<std::mutex> lock(lock_);
    MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " waiting.";
    condition_var_.wait(lock, [this] { return ready_; });
    MS_LOG(DEBUG) << this << " received notify and wake up: " << ready_ << " thread id:" << thread_id_;
    ProcessResult();
    auto ans = abstract_ptr_->TryGetResult();
    MS_EXCEPTION_IF_NULL(ans);
    MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " active.";
    return ans;
  }

  void SetReady() {
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 0b001;  // Set the first bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_ << " result: " << abstract_ptr_->TryGetResult().get()
                    << " thread_id: " << thread_id_;
    }
    condition_var_.notify_one();
  }

  void SetException() {
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 0b010;  // Set the second bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_;
    }
    condition_var_.notify_one();
  }

  void SetEndLessLoopException() {
    {
      std::lock_guard<std::mutex> lock(lock_);
      ready_ = ready_ | 0b100;  // Set the third bit = 1
      MS_LOG(DEBUG) << this << " notify ready: " << ready_;
    }
    condition_var_.notify_one();
  }

 private:
  void HandleEndLessLoopException() {
    // Get third bit
    if (ready_ & 0b100) {
      ready_ = ready_ & 0b011;  // Set the third bit = 0 , Only trigger once.
      MS_LOG(EXCEPTION) << "There isn't any branch that can be evaluated. \n"
                        << "Please check the code if it has the infinite recursion or loop.\n"
                        << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
    }
  }
  void ProcessResult() {
    HandleEndLessLoopException();
    StaticAnalysisException::Instance().CheckException();
    MS_LOG(DEBUG) << this << " Success to GetResult. ready: " << ready_ << " thread_id: " << thread_id_
                  << " result: " << abstract_ptr_->TryGetResult().get();
  }
  std::string thread_id_;
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
  const EvalResultCache &GetCache() const { return eval_result_cache_; }
  EvalResultPtr GetValue(const AbstractBasePtrList &key) { return eval_result_cache_.get(key); }
  void SetValue(const AbstractBasePtrList &key, const EvalResultPtr &arg) { eval_result_cache_.set(key, arg); }
  size_t GetSize() const { return eval_result_cache_.size(); }

 private:
  EvalResultCache eval_result_cache_;
};

// AnalysisCache
class AnalysisResultCacheMgr {
 public:
  using AnalysisConfigResultMap =
    mindspore::HashMap<AnfNodeConfigPtr, EvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigResultCache = NormalCache<AnfNodeConfigPtr, EvalResultPtr, AnalysisConfigResultMap>;
  using const_iterator = typename AnalysisConfigResultCache::const_iterator;

  ~AnalysisResultCacheMgr() = default;
  AnalysisResultCacheMgr(const AnalysisResultCacheMgr &) = delete;
  AnalysisResultCacheMgr &operator=(const AnalysisResultCacheMgr &) = delete;
  static AnalysisResultCacheMgr &GetInstance() {
    static AnalysisResultCacheMgr instance;
    return instance;
  }
  void Clear();
  const AnalysisConfigResultCache &GetCache() const { return cache_; }
  inline void SetValue(const AnfNodeConfigPtr &conf, const EvalResultPtr &arg) { cache_.set(conf, arg); }
  inline EvalResultPtr GetValue(const AnfNodeConfigPtr &conf) { return cache_.get(conf); }
  void InitSwitchValue(const AnfNodeConfigPtr &conf);
  AbstractBasePtr GetSwitchValue(const AnfNodeConfigPtr &conf);
  void SetSwitchValue(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg);
  const_iterator begin() { return cache_.begin(); }
  const_iterator end() { return cache_.end(); }
  void CheckSwitchValueJoinable(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg);
  const PrimitiveEvalCachePtr &prim_eval_cache() const { return prim_eval_cache_; }

 private:
  using AnalysisConfigAsyncResultMap =
    mindspore::HashMap<AnfNodeConfigPtr, AsyncAbstractPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;
  using AnalysisConfigAsyncResultCache =
    MultiThreadCache<AnfNodeConfigPtr, AsyncAbstractPtr, AnalysisConfigAsyncResultMap>;
  AnalysisResultCacheMgr() = default;
  void SetCacheValue(const AnfNodeConfigPtr &conf, const AbstractBasePtr &current_abs,
                     AnalysisConfigAsyncResultCache *cache);

  std::mutex lock_;
  AnalysisConfigResultCache cache_;
  AnalysisConfigAsyncResultCache switch_cache_;
  AnalysisConfigAsyncResultCache switch_cache_for_check_;
  PrimitiveEvalCachePtr prim_eval_cache_ = std::make_shared<PrimitiveEvalCache>();
};

std::string ArgsToString(const AbstractBasePtrList &args_abs_list);
bool enable_waiting_branch_eval();
bool NeedWaitForBranches(const AbstractBasePtr &abstract);

inline std::string GetInferThread() { return std::string(" INFER:") + AnalysisSchedule::thread_id() + ":"; }
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_ASYNC_EVAL_RESULT_H_
