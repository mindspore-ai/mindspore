/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_FUTURE_H_
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_FUTURE_H_

#include <memory>
#include <utility>
#include <future>
#include <iostream>
#include <list>
#include "actor/actor.h"
#include "actor/log.h"
#include "async/spinlock.h"
#include "async/status.h"
#include "async/uuid_generator.h"
#include "async/future_base.h"
#include "mindrt/include/mindrt.hpp"

namespace mindspore {
template <typename T>
class Promise;

template <typename T>
class Option;

template <typename T>
class Future : public FutureBase {
 public:
  typedef MindrtStatus WaitForStatus;
  typedef typename FutureData<T>::CompleteCallback CompleteCallback;
  typedef typename FutureData<T>::AbandonedCallback AbandonedCallback;
  typedef FutureData<T> Data;
  Future() : data(new (std::nothrow) Data()) {
    MINDRT_OOM_EXIT(data);
    data->abandoned = true;
  }

  Future(const Future<T> &f) : FutureBase(f), data(f.data) {}

  Future(Future<T> &&f) : data(std::move(f.data)) {}

  explicit Future(const T &t) : data(new (std::nothrow) Data()) {
    MINDRT_OOM_EXIT(data);
    SetValue(std::move(t));
  }

  template <typename V>
  explicit Future(const V &value) : data(new (std::nothrow) Data()) {
    MINDRT_OOM_EXIT(data);
    SetValue(value);
  }

  explicit Future(const MindrtStatus &s) : data(new (std::nothrow) Data()) {
    MINDRT_OOM_EXIT(data);
    SetFailed(s.GetCode());
  }

  explicit Future(const std::shared_ptr<Data> &t) : data(t) {}

  ~Future() override {}

  Future<T> &operator=(const Future<T> &f) {
    if (&f != this) {
      data = f.data;
    }
    return *this;
  }

  bool operator==(const Future<T> &f) const { return data == f.data; }

  bool operator!=(const Future<T> &f) const { return !(*this == f); }

  const T &Get() const {
    if (data->status.IsError()) {
      MS_LOG(WARNING) << "Future::Get() but status == Error: " << GetErrorCode();
      return data->t;
    }

    if (data->gotten) {
      return data->t;
    }

    //        try {
    data->t = data->future.get();
    data->gotten = true;
    //        } catch (std::future_error const &e) {
    //            ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_MINDRT_LOG, "Future error: %s",
    //            "%s",
    //                                e.what());
    //        } catch (std::exception const &e) {
    //            ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_MINDRT_LOG, "Standard exception:
    //            %s",
    //                                "%s", e.what());
    //        } catch (...) {
    //            ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_MINDRT_LOG, "Unknown exception.");
    //        }

    return data->t;
  }

  bool Valid() const noexcept { return data->future.valid(); }

  bool IsInit() const { return data->status.IsInit(); }

  bool IsOK() const { return data->status.IsOK(); }

  bool IsError() const { return data->status.IsError(); }

  MindrtStatus GetStatus() const { return data->status; }

  int32_t GetErrorCode() const {
    const MindrtStatus &status_ = data->status;
    if (status_.IsError()) {
      return status_.GetCode();
    }

    return 0;
  }

  void Wait() const {
    if (!data->status.IsInit()) {
      return;
    }

    data->future.wait();
  }

  template <typename F>
  const Future<T> &OnComplete(internal::DeferredHelper<F> &&deferred) const {
    return OnComplete(std::move(deferred).operator std::function<void(const Future<T> &)>());
  }

  template <typename F>
  const Future<T> &OnAbandoned(internal::DeferredHelper<F> &&deferred) const {
    return OnAbandoned(std::move(deferred).operator std::function<void(const Future<T> &)>());
  }

  const Future<T> &OnComplete(CompleteCallback &&callback) const {
    bool call = false;

    data->lock.lock();
    if (data->status.IsInit()) {
      // using move to make callback execute once
      data->onCompleteCallbacks.push_back(std::move(callback));
    } else {
      call = true;
    }
    data->lock.unlock();

    if (call) {
      std::move(callback)(*this);
    }

    return *this;
  }

  const Future<T> &OnAbandoned(AbandonedCallback &&callback) const {
    bool call = false;

    data->lock.lock();
    if (data->abandoned) {
      call = true;
    } else if (data->status.IsInit()) {
      // using move to make callback execute once
      data->onAbandonedCallbacks.push_back(std::move(callback));
    }
    data->lock.unlock();

    if (call) {
      std::move(callback)(*this);
    }

    return *this;
  }

  void SetValue(T &&t) const { return Set(std::move(t)); }

  void SetValue(const T &t) const { return Set(t); }

  void SetOK() const {
    bool call = false;

    data->lock.lock();
    if (data->status.IsInit()) {
      data->status.SetOK();
      data->promise.set_value(T());
      call = true;
    }
    data->lock.unlock();

    if (call) {
      RunCallbacks();
    }
  }

  void SetFailed(int32_t errCode) const {
    MINDRT_ASSERT(errCode != MindrtStatus::KINIT && errCode != MindrtStatus::KOK);

    bool call = false;

    data->lock.lock();
    if (data->status.IsInit()) {
      data->status.SetCode(errCode);
      data->promise.set_value(T());
      call = true;
    }
    data->lock.unlock();

    if (call) {
      RunCallbacks();
    }
  }

  // remove all callbacks
  void Clear() const { data->Clear(); }

  void Abandon(bool abandon = false) const {
    bool call = false;

    std::list<AbandonedCallback> callbacks;
    data->lock.lock();
    if (!data->abandoned && data->status.IsInit() && (!data->associated || abandon)) {
      call = data->abandoned = true;
      callbacks.swap(data->onAbandonedCallbacks);
    }
    data->lock.unlock();

    if (call) {
      internal::Run(std::move(callbacks), *this);
    }
  }

  template <typename R>
  Future<R> Then(const std::function<Future<R>(const T &)> &f) const {
    std::shared_ptr<Promise<R>> promise(new (std::nothrow) Promise<R>());
    MINDRT_OOM_EXIT(promise);
    Future<R> future = promise->GetFuture();

    std::function<void(const Future<T> &)> handler =
      std::bind(&internal::Thenf<T, R>, f, promise, std::placeholders::_1);

    OnComplete(std::move(handler));

    return future;
  }

  template <typename R>
  Future<R> Then(const std::function<R(const T &)> &f) const {
    std::shared_ptr<Promise<R>> promise(new (std::nothrow) Promise<R>());
    MINDRT_OOM_EXIT(promise);
    Future<R> future = promise->GetFuture();

    std::function<void(const Future<T> &)> handler =
      std::bind(&internal::Then<T, R>, f, promise, std::placeholders::_1);

    OnComplete(std::move(handler));

    return future;
  }

  template <typename R>
  Future<R> Then(const std::function<Future<R>()> &f) const {
    return Then(std::function<Future<R>(const T &)>(std::bind(f)));
  }

  template <typename R>
  Future<R> Then(const std::function<R()> &f) const {
    return Then(std::function<R(const T &)>(std::bind(f)));
  }

  template <typename F>
  auto Then(F &&f) const -> decltype(this->Then(std::forward<F>(f), FutureBase())) {
    return Then(std::forward<F>(f), FutureBase());
  }

  template <typename F>
  const Future<T> &OnComplete(F &&f) const {
    return OnComplete(std::forward<F>(f), FutureBase());
  }

  template <typename F>
  const Future<T> &OnAbandoned(F &&f) const {
    return OnAbandoned(std::forward<F>(f), FutureBase());
  }

  template <typename F, typename R = typename internal::Unwrap<typename std::result_of<F(const T &)>::type>::type>
  Future<R> Then(internal::DeferredHelper<F> &&f, FutureBase) const {
    return Then<R>(std::move(f).operator std::function<Future<R>(const T &)>());
  }

 private:
  template <typename F, typename R = typename internal::Unwrap<typename std::result_of<typename std::enable_if<
                          !std::is_bind_expression<typename std::decay<F>::type>::value, F>::type()>::type>::type>
  Future<R> Then(internal::DeferredHelper<F> &&f, LessFuture) const {
    return Then<R>(std::move(f).operator std::function<Future<R>()>());
  }

  template <typename F, typename R = typename internal::Unwrap<typename std::result_of<F(const T &)>::type>::type>
  Future<R> Then(F &&f, FutureBase) const {
    return Then<R>(std::function<Future<R>(const T &)>(f));
  }

  template <typename F, typename R = typename internal::Unwrap<typename std::result_of<typename std::enable_if<
                          !std::is_bind_expression<typename std::decay<F>::type>::value, F>::type()>::type>::type>
  Future<R> Then(F &&f, LessFuture) const {
    return Then<R>(std::function<Future<R>()>(std::forward<F>(f)));
  }

  template <typename F, typename = typename std::result_of<F(const Future<T> &)>::type>
  const Future<T> &OnComplete(F &&f, FutureBase) const {
    return OnComplete(
      std::function<void(const Future<T> &)>([=](const Future<T> &future) mutable { std::forward<F>(f)(future); }));
  }

  template <typename F, typename = typename std::result_of<typename std::enable_if<
                          !std::is_bind_expression<typename std::decay<F>::type>::value, F>::type()>::type>
  const Future<T> &OnComplete(F &&f, LessFuture) const {
    return OnComplete(std::function<void(const Future<T> &)>([=](const Future<T> &) mutable { std::forward<F>(f)(); }));
  }

  template <typename F, typename = typename std::result_of<F(const Future<T> &)>::type>
  const Future<T> &OnAbandoned(F &&f, FutureBase) const {
    return OnAbandoned(
      std::function<void(const Future<T> &)>([=](const Future<T> &future) mutable { std::forward<F>(f)(future); }));
  }

  template <typename F, typename = typename std::result_of<typename std::enable_if<
                          !std::is_bind_expression<typename std::decay<F>::type>::value, F>::type()>::type>
  const Future<T> &OnAbandoned(F &&f, LessFuture) const {
    return OnAbandoned(
      std::function<void(const Future<T> &)>([=](const Future<T> &) mutable { std::forward<F>(f)(); }));
  }

  void RunCallbacks() const {
    std::shared_ptr<typename Future<T>::Data> copy = data;
    internal::Run(std::move(copy->onCompleteCallbacks), Future<T>(copy));
    copy->Clear();
  }

  void Run() const {
    auto iter = data->onCompleteCallbacks.begin();
    for (; iter != data->onCompleteCallbacks.end(); ++iter) {
      (*iter)(*this);
    }
  }

  template <typename V>
  void Set(V &&value) const {
    bool call = false;

    data->lock.lock();
    if (data->status.IsInit()) {
      data->status.SetOK();
      data->promise.set_value(std::forward<V>(value));
      call = true;
    }
    data->lock.unlock();

    if (call) {
      RunCallbacks();
    }
  }

  template <typename V>
  friend class Future;
  friend class Promise<T>;

  std::shared_ptr<Data> data;
};

template <typename T>
class Promise {
 public:
  Promise() : future() { future.data->abandoned = false; }

  explicit Promise(const T &t) : future(t) {}

  virtual ~Promise() {
    //        try {
    if (future.data) {
      future.Abandon();
    }
    //        } catch (...) {
    //        }
  }

  void SetValue(const T &value) const { Set(value); }

  void SetValue(T &&value) const { Set(std::move(value)); }

  void SetValue(const Future<T> &tFuture) const { Associate(tFuture); }

  void SetFailed(int32_t code) const {
    if (!future.data->associated) {
      future.SetFailed(code);
    }
  }

  Future<T> GetFuture() const { return future; }

  void Associate(const Future<T> &f) const {
    bool associated = false;

    future.data->lock.lock();
    if (future.IsInit() && !future.data->associated) {
      associated = (future.data->associated = true);
    }
    future.data->lock.unlock();

    if (associated) {
      f.OnComplete(std::bind(&internal::Complete<T>, future, std::placeholders::_1))
        .OnAbandoned(std::bind(&internal::Abandon<T>, future, true));
    }
  }

 private:
  template <typename V>
  void Set(V &&value) const {
    if (future.IsInit() && !future.data->associated) {
      future.SetValue(std::forward<V>(value));
    }
  }

  template <typename V>
  friend class Future;

  Future<T> future;
};

template <>
class Promise<void>;

template <typename T>
class Promise<T &>;
};  // namespace mindspore

#endif
