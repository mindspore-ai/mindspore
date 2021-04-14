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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_COLLECT_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_COLLECT_H

#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <tuple>
#include "async/common.h"
#include "async/future.h"
#include "async/defer.h"
#include "async/spinlock.h"
#include "actor/actor.h"
#include "litebus.hpp"

namespace mindspore {
template <typename T>
class Future;

template <typename T>
class Promise;

template <typename T>
class Collected;

template <typename T>
class Collected {
 public:
  Collected(const std::list<Future<T>> &f, Promise<std::list<T>> *p) : futures(f), promise(p), ready(0) {}

  virtual ~Collected() {
    delete promise;
    promise = nullptr;
  }

  Collected(const Collected &) = delete;
  Collected(Collected &&) = default;

  Collected &operator=(const Collected &) = delete;
  Collected &operator=(Collected &&) = default;

 public:
  void Discarded() {
    auto iter = futures.begin();
    for (; iter != futures.end(); ++iter) {
      iter->SetFailed(MindrtStatus::KERROR);
    }
  }

  void Waited(const Future<T> &future) {
    if (future.IsError()) {
      promise->SetFailed(future.GetErrorCode());
    } else if (future.IsOK()) {
      ready.fetch_add(1);
      if (ready.load() == futures.size()) {
        std::list<T> values;
        auto iter = futures.begin();
        for (; iter != futures.end(); ++iter) {
          values.push_back(iter->Get());
        }
        promise->SetValue(values);
      }
    }
  }

 private:
  const std::list<Future<T>> futures;
  Promise<std::list<T>> *promise;
  std::atomic_ulong ready;
};

template <typename T>
inline Future<std::list<T>> Collect(const std::list<Future<T>> &futures) {
  if (futures.empty()) {
    return Future(std::list<T>());
  }

  Promise<std::list<T>> *promise = new (std::nothrow) Promise<std::list<T>>();
  BUS_OOM_EXIT(promise);
  using CollectType = Collected<T>;
  std::shared_ptr<CollectType> collect = std::make_shared<CollectType>(futures, promise);

  //
  auto iter = futures.begin();
  for (; iter != futures.end(); ++iter) {
    iter->OnComplete(Defer(collect, &CollectType::Waited, std::placeholders::_1));
  }

  Future<std::list<T>> future = promise->GetFuture();
  future.OnComplete(Defer(collect, &Collected<T>::Discarded));

  return future;
}

template <typename... Ts>
Future<std::tuple<Ts...>> Collect(const Future<Ts> &... futures) {
  std::list<Future<Nothing>> wrappers = {futures.Then([]() { return Nothing(); })...};

  auto f = [](const Future<Ts> &... futures) { return std::make_tuple(futures.Get()...); };

  return Collect(wrappers).Then(std::bind(f, futures...));
}
};  // namespace mindspore
#endif
