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

#include "pipeline/jit/static_analysis/async_eval_result.h"
#include <chrono>
#include "utils/symbolic.h"
#include "debug/common.h"
#include "pipeline/jit/base.h"
#include "utils/utils.h"
#include "abstract/utils.h"

namespace mindspore {
namespace abstract {
EvalResultPtr AsyncEvalResult::TryGetResult(int ms) {
  if (result_ != nullptr || ms == 0) {
    return result_;
  }
  std::unique_lock<std::mutex> lock(lock_);
  auto time = std::chrono::microseconds(ms);
  // Wait for ms.
  (void)condition_var_.wait_for(lock, time, [this] { return result_ != nullptr; });
  return result_;
}

EvalResultPtr AsyncEvalResult::GetResult() {
  if (result_ != nullptr) {
    return result_;
  }
  std::unique_lock<std::mutex> lock(lock_);
  auto time = std::chrono::seconds(kInferTimeout);
  (void)condition_var_.wait_for(lock, time, [this] { return result_ != nullptr; });
  return result_;
}

std::string AsyncEvalResult::ToString() {
  std::ostringstream buffer;
  std::lock_guard<std::mutex> lock(lock_);
  buffer << (result_ == nullptr ? "NOT SET" : result_->abstract()->ToString());
  return buffer.str();
}

void AsyncEvalResult::JoinResult(const EvalResultPtr &result) {
  MS_EXCEPTION_IF_NULL(result);
  {
    std::lock_guard<std::mutex> lock(lock_);
    result_ = result;
  }
  condition_var_.notify_all();
}

void AnalysisResultCacheMgr::Clear() {
  cache_.clear();
  switch_cache_.clear();
  todo_.clear();
}

AnalysisResultCacheMgr &AnalysisResultCacheMgr::GetInstance() {
  static AnalysisResultCacheMgr instance;
  return instance;
}

void AnalysisResultCacheMgr::DumpCache(const std::string &filename) {
  auto path = pipeline::GetSaveGraphsPathName(Common::AddId(filename, ".cache"));

  auto realpath = Common::GetRealPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << path;
    return;
  }
  ChangeFileMode(realpath.value(), S_IRWXU);
  std::ofstream fout(realpath.value());
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!";
    return;
  }
  fout << cache_.dump();
  fout.close();
  // Set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}

thread_local static std::string local_threadid;
void AnalysisResultCacheMgr::UpdateCaller(const std::string &caller) {
  std::ostringstream buffer;
  buffer << caller << "." << std::this_thread::get_id();
  local_threadid = buffer.str();
}
std::mutex AnalysisResultCacheMgr::tiggerToken_;
std::string &AnalysisResultCacheMgr::GetThreadid() { return local_threadid; }

void AnalysisResultCacheMgr::PushTowait(const std::shared_future<EvalResultPtr> &future0,
                                        const std::shared_future<EvalResultPtr> &future1) {
  std::lock_guard<std::recursive_mutex> lock(lock_);
  waiting_.push_back(future0);
  waiting_.push_back(future1);
}

void AnalysisResultCacheMgr::PushTodo(const AnfNodeConfigPtr &conf) {
  std::lock_guard<std::recursive_mutex> lock(lock_);
  todo_.push_back(conf);
}

void AnalysisResultCacheMgr::InitSwitchValue(const AnfNodeConfigPtr &conf) {
  std::lock_guard<std::recursive_mutex> lock(lock_);
  AsyncEvalResultPtr async_eval_result = switch_cache_.get(conf);
  if (async_eval_result == nullptr) {
    async_eval_result = std::make_shared<AsyncEvalResult>();
    switch_cache_.set(conf, async_eval_result);
  }
}

EvalResultPtr AnalysisResultCacheMgr::GetSwitchValue(const AnfNodeConfigPtr &conf) {
  AsyncEvalResultPtr async_eval_result = switch_cache_.get(conf);
  // Conf has been visited and set value.
  if (async_eval_result != nullptr) {
    // Maybe blocked for waiting. AsyncEvalResult maybe null, if time out.
    auto result = async_eval_result->GetResult();
    if (result == nullptr) {
      result = std::make_shared<EvalResult>(std::make_shared<AbstractTimeOut>(), nullptr);
      MS_LOG(ERROR) << "AsyncEvalResult for NodeConfig " << conf->ToString() << " is nullptr, maybe timeout.";
    }
    return result;
  }
  return nullptr;
}

void AnalysisResultCacheMgr::SetSwitchValue(const AnfNodeConfigPtr &conf, const EvalResultPtr arg) {
  MS_EXCEPTION_IF_NULL(conf);
  if (arg == nullptr || arg->abstract() == nullptr) {
    MS_LOG(WARNING) << conf->ToString() << " value is nullptr";
  }
  std::lock_guard<std::recursive_mutex> lock(lock_);
  AsyncEvalResultPtr async_eval_result = switch_cache_.get(conf);
  if (async_eval_result == nullptr) {
    MS_LOG(EXCEPTION) << conf->ToString() << " Not key.";
    async_eval_result = std::make_shared<AsyncEvalResult>();
    async_eval_result->JoinResult(arg);
    switch_cache_.set(conf, async_eval_result);
  } else {
    auto ab1 = async_eval_result->TryGetResult();
    AbstractBasePtrList absList;
    if (ab1 != nullptr) {
      absList.push_back(arg->abstract());
      absList.push_back(ab1->abstract());
      // Join two branches's result
      auto joined_spec = AbstractJoin(absList);
      MS_EXCEPTION_IF_NULL(joined_spec);
      MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_spec->ToString();
      auto joined_result = std::make_shared<EvalResult>(joined_spec, std::make_shared<AttrValueMap>());
      async_eval_result->JoinResult(joined_result);
      if (joined_result != ab1) {
        PushTodo(conf);
      }
    } else {
      async_eval_result->JoinResult(arg);
    }
  }
}

void AnalysisResultCacheMgr::Todo() {
  while (true) {
    AnfNodeConfigPtr conf;
    lock_.lock();
    if (!todo_.empty()) {
      conf = todo_.front();
    } else {
      lock_.unlock();
      break;
    }
    todo_.pop_front();
    lock_.unlock();
    if (!(*GetValue(conf)->abstract() == *GetSwitchValue(conf)->abstract())) {
      MS_LOG(WARNING) << " Switch Value is not eq. "
                      << " switchCache: " << GetSwitchValue(conf)->abstract()->ToString()
                      << " globleCache: " << GetValue(conf)->abstract()->ToString() << "\t\tConf: " << conf->ToString();
    }
  }
}

void AnalysisResultCacheMgr::Wait() {
  while (true) {
    std::shared_future<EvalResultPtr> future;
    lock_.lock();
    if (!waiting_.empty()) {
      future = std::move(waiting_.front());
    } else {
      lock_.unlock();
      break;
    }

    waiting_.pop_front();
    lock_.unlock();
    future.wait();
  }
  if (IS_OUTPUT_ON(DEBUG)) {
    Todo();
  }
}

std::string ArgsToString(const AbstractBasePtrList &args_spec_list) {
  std::ostringstream buffer;
  buffer << "(";
  for (const auto &item : args_spec_list) {
    buffer << item->ToString() << " # ";
  }
  buffer << " )";
  return buffer.str();
}
}  // namespace abstract
}  // namespace mindspore
