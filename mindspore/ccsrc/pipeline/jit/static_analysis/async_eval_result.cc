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
#include <debug/trace.h>
#include "utils/symbolic.h"
#include "debug/common.h"
#include "pipeline/jit/base.h"
#include "utils/utils.h"

namespace mindspore {
namespace abstract {
AnalysisSchedule AnalysisSchedule::instance_;

void AnalysisSchedule::Schedule() {
  const auto checkPeriod = std::chrono::seconds(3);
  while (notExit_ || infer_thread_count_.load() > 0) {
    std::unique_lock<std::mutex> lock(activate_thread_lock_);
    if (activate_threads_.size() > 1) {
      MS_LOG(ERROR) << "There is something wrong."
                    << " The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << scheduleList_.size();
    }

    auto ok = activate_thread_cv_.wait_for(lock, checkPeriod, [this] { return activate_threads_.empty(); });
    if (ok && (!SetNextReady())) {
      // If schedule list is empty, wait.
      (void)activate_thread_cv_.wait_for(lock, checkPeriod, [this] { return !scheduleList_.empty(); });
    }
  }
  MS_LOG(DEBUG) << "Success to exit. The active thread count: " << activate_threads_.size()
                << " The infer_thread_count: " << infer_thread_count_
                << " schedule list size: " << scheduleList_.size();
}

void AnalysisSchedule::Yield(const AsyncInferTask *async_infer_task) {
  {
    std::lock_guard<std::mutex> activeLock(activate_thread_lock_);
    // Double check ready()
    if (async_infer_task->Ready() == 0) {
      MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size() << " thread id: " << GetThreadID()
                    << " async_infer_task thread id:" << async_infer_task->ThreadID();
      (void)activate_threads_.erase(GetThreadID());
    }
    MS_LOG(DEBUG) << " The active thread count: " << activate_threads_.size()
                  << " The infer_thread_count: " << infer_thread_count_
                  << " schedule list size: " << scheduleList_.size() << " thread: " << GetThreadID() + " "
                  << (activate_threads_.size() > 0 ? activate_threads_.begin()->c_str() : "");
  }
  activate_thread_cv_.notify_one();
}

void AnalysisSchedule::HandleException(const std::exception &ex) {
  // Just record the first exception information.
  if (!StaticAnalysisException::Instance().HasException()) {
    StaticAnalysisException::Instance().SetException();

    // If python Exception, record the eval stack.
    if (dynamic_cast<const py::error_already_set *>(&ex) != nullptr) {
      try {
        MS_LOG(DEBUG) << "Python exception happened, check the information as below.";
        std::ostringstream exceptionStream;
        trace::GetTraceStackInfo(exceptionStream);
        if (!exceptionStream.str().empty()) {
          MS_LOG(ERROR) << "Exception happened, check the information as below.\n" << exceptionStream.str();
        }
      } catch (const std::exception &e) {
        // Ignored.
      }
    }
  }
  // Free all the locks. Let all the threads continue to run.
  std::lock_guard<std::mutex> lock(activate_thread_lock_);
  for (auto &item : scheduleList_) {
    item->SetException();
  }
  scheduleList_.clear();
}

void AnalysisSchedule::Wait() {
  EnterWaiting();
  if (infer_thread_count_.load() > 0) {
    py::gil_scoped_release infer_gil_release;
    std::unique_lock<std::mutex> lock(infer_thread_lock_);
    infer_thread_cv_.wait(lock, [this] { return infer_thread_count_.load() <= 0; });
  }
  if (infer_thread_count_.load() < 0) {
    MS_LOG(ERROR) << "There is something wrong. thread count: " << infer_thread_count_;
  }
  if (IS_OUTPUT_ON(DEBUG)) {
    AnalysisResultCacheMgr::GetInstance().Todo();
  }
  MS_LOG(INFO) << "Infer finished.";
  StaticAnalysisException::Instance().CheckException();
}

void AnalysisSchedule::Add2Schedule(const AsyncInferTaskPtr &async_infer_task_ptr) {
  std::lock_guard<std::mutex> lock(activate_thread_lock_);
  MS_EXCEPTION_IF_NULL(async_infer_task_ptr);
  scheduleList_.push_back(async_infer_task_ptr);
  activate_thread_cv_.notify_one();
  MS_LOG(DEBUG) << " async: " << async_infer_task_ptr->ThreadID() << " address: " << async_infer_task_ptr.get()
                << " The active thread count: " << activate_threads_.size()
                << " The infer_thread_count: " << infer_thread_count_
                << " schedule list size: " << scheduleList_.size();
}
bool AnalysisSchedule::SetNextReady() {
  if (scheduleList_.empty()) {
    MS_LOG(DEBUG) << "The schedule list is empty. ";
    return false;
  }
  // Check if enter endless loop
  auto it = std::find_if(scheduleList_.begin(), scheduleList_.end(), [](const auto &item) {
    MS_EXCEPTION_IF_NULL(item);
    return item->HasResult();
  });
  if (it == scheduleList_.end()) {
    if (IntToSize(infer_thread_count_.load()) >= scheduleList_.size()) {
      MS_LOG(DEBUG) << "There is some task to be added. Please wait.";
      return false;
    }
    MS_LOG(WARNING) << "Enter endless loop. The active thread count: " << activate_threads_.size()
                    << " The infer_thread_count: " << infer_thread_count_
                    << " schedule list size: " << scheduleList_.size();
    // Enter endless loop if there is not ready result.
    (void)activate_threads_.insert(scheduleList_.front()->ThreadID());
    // Let the first thread to trigger endless loop exception.
    MS_LOG(DEBUG) << "Enter endless loop if there is not ready result.Set the async to trigger exception:"
                  << scheduleList_.front().get() << " The active thread count: " << activate_threads_.size();
    scheduleList_.front()->SetEndLessLoopException();
    scheduleList_.pop_front();
    return true;
  }
  auto async_task = *it;
  (void)activate_threads_.insert(async_task->ThreadID());
  async_task->SetReady();
  (void)scheduleList_.erase(it);
  MS_LOG(DEBUG) << " Success to SetReady. The active thread count: " << activate_threads_.size()
                << " The infer_thread_count: " << infer_thread_count_ << " schedule list size: " << scheduleList_.size()
                << " async: " << async_task->ThreadID() << "  address: " << async_task.get();

  return true;
}
// The thread id format is XXXX.YYYY.ZZZZ
thread_local std::string localThreadID = "1";
void AnalysisSchedule::SetThreadID(const std::string &threadID) { localThreadID = threadID; }

std::string &AnalysisSchedule::GetThreadID() { return localThreadID; }

AnalysisResultCacheMgr AnalysisResultCacheMgr::instance_;

void AnalysisResultCacheMgr::Clear() {
  std::lock_guard<std::mutex> lock(lock_);
  cache_.clear();
  switch_cache_.clear();
  todo_.clear();
}

void AnalysisResultCacheMgr::PushTodo(const AnfNodeConfigPtr &conf) {
  std::lock_guard<std::mutex> lock(todo_lock_);
  todo_.push_back(conf);
}

void AnalysisResultCacheMgr::InitSwitchValue(const AnfNodeConfigPtr &conf) {
  std::lock_guard<std::mutex> lock(lock_);
  AsyncAbstractPtr async_eval_result = switch_cache_.get(conf);
  if (async_eval_result == nullptr) {
    async_eval_result = std::make_shared<AsyncAbstract>();
    switch_cache_.set(conf, async_eval_result);
  }
}

AbstractBasePtr AnalysisResultCacheMgr::TryGetSwitchValue(const AnfNodeConfigPtr &conf) {
  // don't call lock_.lock(). switch_cache is protected. and it waits for result.
  AsyncAbstractPtr async_eval_result = switch_cache_.get(conf);
  // Conf has been visited and set value.
  if (async_eval_result != nullptr) {
    return async_eval_result->TryGetResult();
  }
  return nullptr;
}

AbstractBasePtr AnalysisResultCacheMgr::GetSwitchValue(const AnfNodeConfigPtr &conf) {
  StaticAnalysisException::Instance().CheckException();
  // don't call lock_.lock(). switch_cache is protected. and it waits for result.
  AsyncAbstractPtr async_eval_result = switch_cache_.get(conf);
  // Conf has been visited and set value.
  if (async_eval_result != nullptr) {
    // Add to schedule
    auto async_infer_task = AsyncInferTask::MakeShared(async_eval_result);
    MS_LOG(DEBUG) << " add to schedule: " << async_infer_task.get();
    AnalysisSchedule::GetInstance().Add2Schedule(async_infer_task);
    // Maybe blocked for waiting. AsyncAbstract maybe null, if time out.
    auto result = async_infer_task->GetResult();
    if (result == nullptr) {
      result = std::make_shared<AbstractTimeOut>();
      MS_LOG(ERROR) << "AsyncAbstract of NodeConfig " << conf->node()->ToString()
                    << " is nullptr. There is something wrong.";
      StaticAnalysisException::Instance().CheckException();
    }
    return result;
  }
  return nullptr;
}

void AnalysisResultCacheMgr::SetSwitchValue(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg) {
  MS_EXCEPTION_IF_NULL(conf);
  if (arg == nullptr) {
    MS_LOG(EXCEPTION) << conf->ToString() << " value is nullptr";
  }
  std::lock_guard<std::mutex> lock(lock_);
  AsyncAbstractPtr async_eval_result = switch_cache_.get(conf);
  if (async_eval_result == nullptr) {
    async_eval_result = std::make_shared<AsyncAbstract>();
    async_eval_result->SetResult(arg);
    switch_cache_.set(conf, async_eval_result);
  } else {
    auto ab1 = async_eval_result->TryGetResult();
    AbstractBasePtrList absList;
    if (ab1 != nullptr) {
      absList.push_back(arg);
      absList.push_back(ab1);
      // Join two branches's result
      auto joined_result = AnalysisEngine::ProcessEvalResults(absList, conf->node());
      async_eval_result->SetResult(joined_result->abstract());
      if (!(*joined_result == *ab1)) {
        PushTodo(conf);
      }
    } else {
      async_eval_result->SetResult(arg);
    }
  }
}

void AnalysisResultCacheMgr::Todo() {
  std::lock_guard<std::mutex> lock(todo_lock_);
  while (!todo_.empty()) {
    AnfNodeConfigPtr conf = todo_.front();
    MS_EXCEPTION_IF_NULL(conf);
    todo_.pop_front();
    if (GetValue(conf) == nullptr) {
      MS_LOG(INFO) << conf->node()->ToString() << " not in globle cache.";
      continue;
    }
    if (TryGetSwitchValue(conf) == nullptr) {
      MS_LOG(INFO) << conf->node()->ToString() << " not in switch cache";
      continue;
    }
    auto switch_value = TryGetSwitchValue(conf);
    auto abstract = GetValue(conf)->abstract();
    MS_EXCEPTION_IF_NULL(switch_value);
    MS_EXCEPTION_IF_NULL(abstract);
    if (!(*abstract == *switch_value)) {
      MS_LOG(WARNING) << " Switch Value is not eq. "
                      << " switchCache: " << switch_value->ToString() << " globleCache: " << abstract->ToString()
                      << "\t\tConf: " << conf->ToString();
    }
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
