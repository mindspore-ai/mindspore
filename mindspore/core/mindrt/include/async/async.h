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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_ASYNC_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_ASYNC_H

#include <tuple>
#include <memory>
#include <utility>

#include "actor/actor.h"
#include "actor/log.h"
#include "actor/actormgr.h"
#include "async/apply.h"
#include "async/future.h"

namespace mindspore {

using MessageHandler = std::function<void(ActorBase *)>;

class MessageAsync : public MessageBase {
 public:
  explicit MessageAsync(MessageHandler &&h) : MessageBase("Async", Type::KASYNC), handler(h) {}
  virtual ~MessageAsync() = default;
  void Run(ActorBase *actor) override { (handler)(actor); }

 private:
  MessageHandler handler;
};

namespace internal {

template <typename R>
struct AsyncHelper;

// for defer
template <>
struct AsyncHelper<void> {
  template <typename F>
  void operator()(const AID &aid, F &&f) {
    std::function<void(ActorBase *)> handler = [=](ActorBase *) { f(); };
    auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
    MINDRT_OOM_EXIT(msg);
    (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  }
};

template <typename R>
struct AsyncHelper<Future<R>> {
  template <typename F>
  Future<R> operator()(const AID &aid, F &&f) {
    auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
    MINDRT_OOM_EXIT(promise);
    Future<R> future = promise->GetFuture();

    MessageHandler handler = [=](ActorBase *) { promise->Associate(f()); };

    auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
    MINDRT_OOM_EXIT(msg);
    (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
    return future;
  }
};

template <typename R>
struct AsyncHelper {
  template <typename F>
  Future<R> operator()(const AID &aid, F &&f) {
    auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
    MINDRT_OOM_EXIT(promise);
    Future<R> future = promise->GetFuture();

    std::function<void(ActorBase *)> handler = [=](ActorBase *) { promise->SetValue(f()); };
    auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
    MINDRT_OOM_EXIT(msg);
    (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
    return future;
  }
};

}  // namespace internal

// return void
template <typename T>
void Async(const AID &aid, void (T::*method)()) {
  std::function<void(ActorBase *)> handler = [method](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    (t->*method)();
  };
  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
}

template <typename T, typename Arg0, typename Arg1>
void Async(const AID &aid, void (T::*method)(Arg0), Arg1 &&arg) {
  std::function<void(ActorBase *)> handler = [method, arg](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    (t->*method)(arg);
  };
  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
}

template <typename T, typename... Args0, typename... Args1>
void Async(const AID &aid, void (T::*method)(Args0...), std::tuple<Args1...> &&tuple) {
  std::function<void(ActorBase *)> handler = [method, tuple](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    Apply(t, method, tuple);
  };
  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
}

template <typename T, typename... Args0, typename... Args1>
void Async(const AID &aid, void (T::*method)(Args0...), Args1 &&... args) {
  auto tuple = std::make_tuple(std::forward<Args1>(args)...);
  Async(aid, method, std::move(tuple));
}

// return future
template <typename R, typename T>
Future<R> Async(const AID &aid, Future<R> (T::*method)()) {
  auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->Associate((t->*method)());
  };
  std::unique_ptr<MessageBase> msg(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename T, typename Arg0, typename Arg1>
Future<R> Async(const AID &aid, Future<R> (T::*method)(Arg0), Arg1 &&arg) {
  auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method, arg](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->Associate((t->*method)(arg));
  };

  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename T, typename... Args0, typename... Args1>
Future<R> Async(const AID &aid, Future<R> (T::*method)(Args0...), std::tuple<Args1...> &&tuple) {
  auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method, tuple](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->Associate(Apply(t, method, tuple));
  };

  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename T, typename... Args0, typename... Args1>
Future<R> Async(const AID &aid, Future<R> (T::*method)(Args0...), Args1 &&... args) {
  auto tuple = std::make_tuple(std::forward<Args1>(args)...);
  return Async(aid, method, std::move(tuple));
}

// return R
template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0,
          typename std::enable_if<!internal::IsFuture<R>::value, int>::type = 0, typename T>
Future<R> Async(const AID &aid, R (T::*method)()) {
  auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->SetValue((t->*method)());
  };

  std::unique_ptr<MessageBase> msg(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0,
          typename std::enable_if<!internal::IsFuture<R>::value, int>::type = 0, typename T, typename Arg0,
          typename Arg1>
Future<R> Async(const AID &aid, R (T::*method)(Arg0), Arg1 &&arg) {
  auto promise = std::shared_ptr<Promise<R>>(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method, arg](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->SetValue((t->*method)(arg));
  };
  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0,
          typename std::enable_if<!internal::IsFuture<R>::value, int>::type = 0, typename T, typename... Args0,
          typename... Args1>
Future<R> Async(const AID &aid, R (T::*method)(Args0...), std::tuple<Args1...> &&tuple) {
  std::shared_ptr<Promise<R>> promise(new (std::nothrow) Promise<R>());
  MINDRT_OOM_EXIT(promise);
  Future<R> future = promise->GetFuture();

  std::function<void(ActorBase *)> handler = [promise, method, tuple](ActorBase *actor) {
    MINDRT_ASSERT(actor != nullptr);
    T *t = static_cast<T *>(actor);
    MINDRT_ASSERT(t != nullptr);
    promise->SetValue(Apply(t, method, tuple));
  };
  auto msg = std::unique_ptr<MessageBase>(new (std::nothrow) MessageAsync(std::move(handler)));
  MINDRT_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
  return future;
}

template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0,
          typename std::enable_if<!internal::IsFuture<R>::value, int>::type = 0, typename T, typename... Args0,
          typename... Args1>
Future<R> Async(const AID &aid, R (T::*method)(Args0...), Args1 &&... args) {
  auto tuple = std::make_tuple(std::forward<Args1>(args)...);
  return Async(aid, method, std::move(tuple));
}

template <typename F, typename R = typename std::result_of<F()>::type>
auto Async(const AID &aid, F &&f) -> decltype(internal::AsyncHelper<R>()(aid, std::forward<F>(f))) {
  return internal::AsyncHelper<R>()(aid, std::forward<F>(f));
}

}  // namespace mindspore

#endif
