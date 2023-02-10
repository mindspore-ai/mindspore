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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_FUTURE_BASE_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_FUTURE_BASE_H

#include <future>
#include <iostream>
#include <utility>
#include <memory>
#include <list>

#include "actor/actor.h"
#include "async/spinlock.h"
#include "async/status.h"

namespace mindspore {
template <typename T>
class Future;

template <typename T>
class Promise;
class LessFuture {
 public:
  LessFuture() {}
  LessFuture(const LessFuture &obj) {}
  LessFuture &operator=(const LessFuture &) = delete;
  virtual ~LessFuture() {}
};

class FutureBase : public LessFuture {
 public:
  FutureBase() {}
  FutureBase(const FutureBase &obj) : LessFuture(obj) {}
  FutureBase &operator=(const FutureBase &) = delete;
  ~FutureBase() override {}
};

template <typename T>
struct FutureData {
 public:
  typedef std::function<void(const Future<T> &)> CompleteCallback;
  typedef std::function<void(const Future<T> &)> AbandonedCallback;

  FutureData()
      : status(MindrtStatus::KINIT),
        associated(false),
        abandoned(false),
        gotten(false),
        promise(),
        future(promise.get_future()),
        t() {}

  ~FutureData() {
    //        try {
    Clear();
    //        } catch (...) {
    //        }
  }

  // remove all callbacks
  void Clear() {
    onCompleteCallbacks.clear();
    onAbandonedCallbacks.clear();
  }

  // status of future
  SpinLock lock;
  MindrtStatus status;

  bool associated;
  bool abandoned;
  bool gotten;

  std::promise<T> promise;

  // get from promise
  std::future<T> future;

  // complete callback
  std::list<CompleteCallback> onCompleteCallbacks;

  // abandoned callback
  std::list<AbandonedCallback> onAbandonedCallbacks;

  T t;
};

namespace internal {
template <typename T>
class DeferredHelper;

template <typename T>
struct Wrap {
  typedef Future<T> type;
};

template <typename T>
struct Wrap<Future<T>> {
  typedef Future<T> type;
};

template <typename T>
struct Unwrap {
  typedef T type;
};

template <typename T>
struct Unwrap<Future<T>> {
  typedef T type;
};

template <typename T>
struct IsFuture : public std::integral_constant<bool, std::is_base_of<FutureBase, T>::value> {};

template <typename H, typename... Args>
static void Run(std::list<H> &&handlers, Args &&... args) {
  for (auto iter = handlers.begin(); iter != handlers.end(); ++iter) {
    std::move (*iter)(std::forward<Args>(args)...);
  }
}

template <typename T>
static void Complete(const Future<T> &future, const Future<T> &f) {
  if (f.IsError()) {
    future.SetFailed(f.GetErrorCode());
  } else if (f.IsOK()) {
    future.SetValue(f.Get());
  }
}

template <typename T>
static void Abandon(const Future<T> &future, bool abandon) {
  future.Abandon(abandon);
}

template <typename T, typename R>
static void Thenf(const std::function<Future<R>(const T &)> &function, const std::shared_ptr<Promise<R>> &promise,
                  const Future<T> &f) {
  if (f.IsError()) {
    promise->SetFailed(f.GetErrorCode());
  } else if (f.IsOK()) {
    promise->Associate(function(f.Get()));
  }
}

template <typename T, typename R>
static void Then(const std::function<R(const T &)> &function, const std::shared_ptr<Promise<R>> &promise,
                 const Future<T> &f) {
  if (f.IsError()) {
    promise->SetFailed(f.GetErrorCode());
  } else if (f.IsOK()) {
    promise->SetValue(function(f.Get()));
  }
}

template <typename T>
static void Afterf(const std::function<Future<T>(const Future<T> &)> &f, const std::shared_ptr<Promise<T>> &promise,
                   const Future<T> &future) {
  promise->Associate(f(future));
}

void Waitf(const AID &aid);
}  // namespace internal
}  // namespace mindspore

#endif
