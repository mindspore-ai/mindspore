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

#ifndef MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#define MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#include <exception>
#include <set>
#include <mutex>
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"
#include "mindapi/base/macros.h"
namespace mindspore {
class ExceptionListener {
 public:
  virtual void OnException() = 0;
  virtual ~ExceptionListener() = default;
};

class MS_CORE_API MsException {
 public:
  static MsException &Instance();

  void SetException() {
    exception_ptr_ = std::current_exception();
    if (exception_ptr_ != nullptr && listener_ != nullptr) {
      auto listener = listener_;
      listener_ = nullptr;
      listener->OnException();
    }
  }

  void CheckException() {
    if (exception_ptr_ != nullptr) {
      auto exception_ptr = exception_ptr_;
      exception_ptr_ = nullptr;
      MS_LOG(DEBUG) << "Find exception and rethrow";
      std::rethrow_exception(exception_ptr);
    }
  }

  void SetExceptionListener(ExceptionListener *listener) { listener_ = listener; }

 private:
  MsException() = default;
  ~MsException() { listener_ = nullptr; }
  DISABLE_COPY_AND_ASSIGN(MsException)
  ExceptionListener *listener_{nullptr};
  std::exception_ptr exception_ptr_{nullptr};
};

class MS_CORE_API StaticAnalysisException {
 public:
  static StaticAnalysisException &Instance();

  void ClearException() {
    std::lock_guard<std::mutex> lock(lock_);
    exception_ptr_ = nullptr;
  }

  bool HasException() {
    std::lock_guard<std::mutex> lock(lock_);
    return exception_ptr_ != nullptr;
  }

  void SetException() {
    std::lock_guard<std::mutex> lock(lock_);
    if (exception_ptr_ != nullptr) {
      return;
    }
    exception_ptr_ = std::current_exception();
  }

  void SetAndRethrowException() {
    std::lock_guard<std::mutex> lock(lock_);
    SetException();
    std::rethrow_exception(std::current_exception());
  }

  void CheckException() {
    std::lock_guard<std::mutex> lock(lock_);
    if (exception_ptr_ != nullptr) {
      auto tmp_exception_ptr = exception_ptr_;
      std::rethrow_exception(tmp_exception_ptr);
    }
  }

 private:
  StaticAnalysisException() = default;
  ~StaticAnalysisException() = default;
  DISABLE_COPY_AND_ASSIGN(StaticAnalysisException)

  std::exception_ptr exception_ptr_{nullptr};
  std::mutex lock_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
