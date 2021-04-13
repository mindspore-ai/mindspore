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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_DEFER_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_DEFER_H

#include <functional>
#include <memory>
#include <utility>

#include "async/async.h"
#include "async/option.h"

namespace mindspore {
template <typename F>
class Deferred : public std::function<F> {
 public:
  virtual ~Deferred() {}

 private:
  template <typename G>
  friend class internal::DeferredHelper;

  template <typename T>
  friend Deferred<void()> Defer(const AID &aid, void (T::*method)());

  template <typename R, typename T>
  friend Deferred<Future<R>()> Defer(const AID &aid, Future<R> (T::*method)());

  template <typename R, typename T>
  friend Deferred<Future<R>()> Defer(const AID &aid, R (T::*method)());

  Deferred(const std::function<F> &f) : std::function<F>(f) {}
};

namespace internal {
template <typename F>
class DeferredHelper {
 public:
  DeferredHelper(const AID &id, F &&function) : aid(id), f(std::forward<F>(function)) {}

  DeferredHelper(F &&function) : f(std::forward<F>(function)) {}

  ~DeferredHelper() {}

  operator Deferred<void()>() && {
    if (aid.IsNone()) {
      return std::function<void()>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void()>([=]() { Async(optionAid.Get(), function); });
  }

  operator std::function<void()>() && {
    if (aid.IsNone()) {
      return std::function<void()>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void()>([=]() { Async(optionAid.Get(), function); });
  }

  template <typename R>
  operator Deferred<R()>() && {
    if (aid.IsNone()) {
      return std::function<R()>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R()>([=]() { return Async(optionAid.Get(), function); });
  }

  template <typename R>
  operator std::function<R()>() && {
    if (aid.IsNone()) {
      return std::function<R()>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R()>([=]() { return Async(optionAid.Get(), function); });
  }

  template <typename Arg>
  operator Deferred<void(Arg)>() && {
    if (aid.IsNone()) {
      return std::function<void(Arg)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void(Arg)>([=](Arg arg) {
      std::function<void()> handler([=]() { function(arg); });
      Async(optionAid.Get(), handler);
    });
  }

  template <typename Arg>
  operator std::function<void(Arg)>() && {
    if (aid.IsNone()) {
      return std::function<void(Arg)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void(Arg)>([=](Arg arg) {
      std::function<void()> handler([=]() { function(arg); });
      Async(optionAid.Get(), handler);
    });
  }

  template <typename... Args>
  operator Deferred<void(Args...)>() && {
    if (aid.IsNone()) {
      return std::function<void(Args...)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void(Args...)>([=](Args... args) {
      auto tuple = std::make_tuple(std::forward<Args>(args)...);
      std::function<void()> handler([=]() { Apply(function, tuple); });
      Async(optionAid.Get(), handler);
    });
  }

  template <typename... Args>
  operator std::function<void(Args...)>() && {
    if (aid.IsNone()) {
      return std::function<void(Args...)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<void(Args...)>([=](Args... args) {
      auto tuple = std::make_tuple(std::forward<Args>(args)...);
      std::function<void()> handler([=]() { Apply(function, tuple); });
      Async(optionAid.Get(), handler);
    });
  }

  template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0, typename Arg>
  operator Deferred<R(Arg)>() && {
    if (aid.IsNone()) {
      return std::function<R(Arg)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R(Arg)>([=](Arg arg) {
      std::function<R()> handler([=]() { return function(arg); });
      return Async(optionAid.Get(), handler);
    });
  }

  template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0, typename Arg>
  operator std::function<R(Arg)>() && {
    if (aid.IsNone()) {
      return std::function<R(Arg)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R(Arg)>([=](Arg arg) {
      std::function<R()> handler([=]() { return function(arg); });
      return Async(optionAid.Get(), handler);
    });
  }

  template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0, typename... Args>
  operator Deferred<R(Args...)>() && {
    if (aid.IsNone()) {
      return std::function<R(Args...)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R(Args...)>([=](Args... args) {
      auto tuple = std::make_tuple(std::forward<Args>(args)...);
      std::function<R()> handler([=]() { return Apply(function, tuple); });
      return Async(optionAid.Get(), handler);
    });
  }

  template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0, typename... Args>
  operator std::function<R(Args...)>() && {
    if (aid.IsNone()) {
      return std::function<R(Args...)>(std::forward<F>(f));
    }

    Option<AID> optionAid = aid;
    F &&function = std::forward<F>(f);

    return std::function<R(Args...)>([=](Args... args) {
      auto tuple = std::make_tuple(std::forward<Args>(args)...);
      std::function<R()> handler([=]() { return Apply(function, tuple); });
      return Async(optionAid.Get(), handler);
    });
  }

 private:
  template <typename G>
  friend DeferredHelper<G> Defer(const AID &aid, G &&g);

  Option<AID> aid;
  F f;
};

}  // namespace internal

template <typename F>
internal::DeferredHelper<F> Defer(const AID &aid, F &&f) {
  return internal::DeferredHelper<F>(aid, std::forward<F>(f));
}

template <typename T>
Deferred<void()> Defer(const AID &aid, void (T::*method)()) {
  return Deferred<void()>([=]() { Async(aid, method); });
}

template <typename R, typename T>
Deferred<Future<R>()> Defer(const AID &aid, Future<R> (T::*method)()) {
  return Deferred<Future<R>()>([=]() { return Async(aid, method); });
}

template <typename R, typename T>
Deferred<Future<R>()> Defer(const AID &aid, R (T::*method)()) {
  return Deferred<Future<R>()>([=]() { return Async(aid, method); });
}

template <typename T, typename... Args0, typename... Args1>
auto Defer(T *t, void (T::*method)(Args0...), Args1 &&... args)
  -> internal::DeferredHelper<decltype(std::bind(&std::function<void(Args0...)>::operator(),
                                                 std::function<void(Args0...)>(), std::forward<Args1>(args)...))> {
  std::function<void(Args0...)> f([=](Args0... args0) {
    if (t != nullptr) {
      (t->*method)(args0...);
    }
  });

  return std::bind(&std::function<void(Args0...)>::operator(), std::move(f), std::forward<Args1>(args)...);
}

template <typename T, typename... Args0, typename... Args1>
auto Defer(std::shared_ptr<T> t, void (T::*method)(Args0...), Args1 &&... args)
  -> internal::DeferredHelper<decltype(std::bind(&std::function<void(Args0...)>::operator(),
                                                 std::function<void(Args0...)>(), std::forward<Args1>(args)...))> {
  std::function<void(Args0...)> f([=](Args0... args0) {
    if (t != nullptr) {
      (t.get()->*method)(args0...);
    }
  });

  return std::bind(&std::function<void(Args0...)>::operator(), std::move(f), std::forward<Args1>(args)...);
}

template <typename T, typename... Args0, typename... Args1>
auto Defer(const AID &aid, void (T::*method)(Args0...), Args1 &&... args)
  -> internal::DeferredHelper<decltype(std::bind(&std::function<void(Args0...)>::operator(),
                                                 std::function<void(Args0...)>(), std::forward<Args1>(args)...))> {
  std::function<void(Args0...)> f([=](Args0... args0) { Async(aid, method, args0...); });

  return std::bind(&std::function<void(Args0...)>::operator(), std::move(f), std::forward<Args1>(args)...);
}

template <typename R, typename T, typename... Args0, typename... Args1>
auto Defer(const AID &aid, Future<R> (T::*method)(Args0...), Args1 &&... args)
  -> internal::DeferredHelper<decltype(std::bind(&std::function<Future<R>(Args0...)>::operator(),
                                                 std::function<Future<R>(Args0...)>(), std::forward<Args1>(args)...))> {
  std::function<Future<R>(Args0...)> f([=](Args0... args0) { return Async(aid, method, args0...); });

  return std::bind(&std::function<Future<R>(Args0...)>::operator(), std::move(f), std::forward<Args1>(args)...);
}

template <typename R, typename std::enable_if<!std::is_same<R, void>::value, int>::type = 0,
          typename std::enable_if<!internal::IsFuture<R>::value, int>::type = 0, typename T, typename... Args0,
          typename... Args1>
auto Defer(const AID &aid, R (T::*method)(Args0...), Args1 &&... args)
  -> internal::DeferredHelper<decltype(std::bind(&std::function<Future<R>(Args0...)>::operator(),
                                                 std::function<Future<R>(Args0...)>(), std::forward<Args1>(args)...))> {
  std::function<Future<R>(Args0...)> f([=](Args0... args0) { return Async(aid, method, args0...); });

  return std::bind(&std::function<Future<R>(Args0...)>::operator(), std::move(f), std::forward<Args1>(args)...);
}
}  // namespace mindspore
#endif
