/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SIGNAL_H_
#define MINDSPORE_CORE_UTILS_SIGNAL_H_

#include <functional>
#include <memory>
#include <vector>
#include <utility>

namespace mindspore {
template <class Return, class Type, class... Args>
std::function<Return(Args...)> bind_member(Type *instance, Return (Type::*method)(Args...)) {
  return [=](Args &&... args) -> Return { return (instance->*method)(std::forward<Args>(args)...); };
}

template <class FuncType>
class Slot {
 public:
  explicit Slot(const std::function<FuncType> &callback) : callback(callback) {}

  ~Slot() {}

  std::function<FuncType> callback = nullptr;
};

template <class FuncType>
class Signal {
 public:
  template <class... Args>
  void operator()(Args &&... args) const {
    for (auto &slot : slots_) {
      if (slot->callback != nullptr) {
        slot->callback(std::forward<Args>(args)...);
      }
    }
  }

  void add_slot(const std::function<FuncType> &func) {
    auto slot = std::make_shared<Slot<FuncType>>(func);
    slots_.push_back(slot);
  }

  // signal connect to a class member func
  template <class InstanceType, class MemberFuncType>
  void connect(InstanceType instance, MemberFuncType func) {
    add_slot(bind_member(instance, func));
  }

 private:
  std::vector<std::shared_ptr<Slot<FuncType>>> slots_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SIGNAL_H_
