/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_COUNTER_H_
#define MINDSPORE_CORE_UTILS_COUNTER_H_
#include <list>
#include <vector>
#include <utility>
#include <functional>
#include <unordered_map>
#include <memory>
#include "utils/ordered_map.h"

namespace mindspore {
template <typename T, class Hash = std::hash<T>, class Equal = std::equal_to<T>>
class Counter {
  using counter_type = Counter<T, Hash, Equal>;
  using key_type = T const *;
  using item_type = std::pair<T, int>;
  using list_type = std::list<item_type>;
  using iterator = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;

  struct KeyHash {
    std::size_t operator()(const key_type ptr) const noexcept { return Hash{}(*ptr); }
  };

  struct KeyEqual {
    bool operator()(const key_type lhs, const key_type rhs) const noexcept { return Equal{}(*lhs, *rhs); }
  };
  using map_type = std::unordered_map<key_type, iterator, KeyHash, KeyEqual>;

 public:
  Counter() = default;
  ~Counter() = default;

  Counter(Counter &&other) noexcept = default;
  Counter &operator=(Counter &&other) noexcept = default;

  Counter(const Counter &other) { *this = other; }
  Counter &operator=(const Counter &other) {
    map_.clear();
    list_ = other.list_;
    for (auto iter = list_.begin(); iter != list_.end(); ++iter) {
      map_.emplace(&(iter->first), iter);
    }
    return *this;
  }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&... args) {
    auto new_iter = list_.emplace(list_.end(), std::forward<Args>(args)...);
    auto [map_iter, inserted] = map_.emplace(&(new_iter->first), new_iter);
    if (!inserted) {
      list_.erase(new_iter);
    }
    return {map_iter->second, inserted};
  }

  template <typename... Args>
  void add(Args &&... args) {
    auto [iter, inserted] = emplace(T{std::forward<Args>(args)...}, 1);
    if (!inserted) {
      ++(iter->second);
    }
  }

  int &operator[](const T &key) {
    auto map_iter = map_.find(&key);
    if (map_iter != map_.end()) {
      return map_iter->second->second;
    }
    return emplace(key, 0).first->second;
  }

  counter_type operator-(const counter_type &other) const {
    counter_type new_counter;
    for (const auto &[key, value] : list_) {
      auto iter = other.find(key);
      if (iter != other.end()) {
        int new_value = value - iter->second;
        if (new_value > 0) {
          new_counter.emplace(key, new_value);
        }
      } else {
        new_counter.emplace(key, value);
      }
    }
    return new_counter;
  }

  counter_type operator+(const counter_type &other) const {
    counter_type new_counter = *this;
    for (const auto &[key, value] : other.list_) {
      auto [iter, inserted] = new_counter.emplace(key, value);
      if (!inserted) {
        iter->second += value;
      }
    }
    return new_counter;
  }

  template <typename Func>
  void subtract_by(const counter_type &other, Func &&func) const {
    for (const auto &[key, value] : list_) {
      auto iter = other.find(key);
      if (iter != other.end()) {
        if ((value - iter->second) > 0) {
          func(key);
        }
      } else {
        func(key);
      }
    }
  }

  std::vector<T> subtract(const counter_type &other) const {
    std::vector<T> result;
    subtract_by(other, [&result](const T &item) { result.emplace_back(item); });
    return result;
  }

  std::size_t size() const { return list_.size(); }

  bool contains(const T &key) const { return map_.find(&key) != map_.end(); }

  const_iterator find(const T &key) const {
    auto map_iter = map_.find(&key);
    if (map_iter == map_.end()) {
      return list_.end();
    }
    return map_iter->second;
  }

  iterator begin() { return list_.begin(); }
  iterator end() { return list_.end(); }

  const_iterator begin() const { return list_.cbegin(); }
  const_iterator end() const { return list_.cend(); }

  const_iterator cbegin() const { return list_.cbegin(); }
  const_iterator cend() const { return list_.cend(); }

 private:
  map_type map_;
  list_type list_;
};

// Counter for shared_ptr.
template <typename T>
class Counter<std::shared_ptr<T>> {
  using key_type = std::shared_ptr<T>;
  using counter_type = Counter<key_type>;
  using map_type = OrderedMap<key_type, int>;
  using item_type = std::pair<std::shared_ptr<T>, int>;
  using iterator = typename map_type::iterator;
  using const_iterator = typename map_type::const_iterator;

 public:
  std::pair<iterator, bool> emplace(const key_type &key, int value) { return map_.emplace(key, value); }

  std::pair<iterator, bool> emplace(key_type &&key, int value) { return map_.emplace(std::move(key), value); }

  void add(const key_type &key) {
    auto [iter, inserted] = map_.emplace(key, 1);
    if (!inserted) {
      ++(iter->second);
    }
  }

  void add(key_type &&key) {
    auto [iter, inserted] = map_.emplace(std::move(key), 1);
    if (!inserted) {
      ++(iter->second);
    }
  }

  int &operator[](const T &key) { return map_[key]; }

  counter_type operator-(const counter_type &other) const {
    counter_type new_counter;
    for (const auto &[key, value] : map_) {
      auto iter = other.find(key);
      if (iter != other.end()) {
        int new_value = value - iter->second;
        if (new_value > 0) {
          new_counter.emplace(key, new_value);
        }
      } else {
        new_counter.emplace(key, value);
      }
    }
    return new_counter;
  }

  counter_type operator+(const counter_type &other) const {
    counter_type new_counter = *this;
    for (const auto &[key, value] : other) {
      auto [iter, inserted] = new_counter.emplace(key, value);
      if (!inserted) {
        iter->second += value;
      }
    }
    return new_counter;
  }

  template <typename Func>
  void subtract_by(const counter_type &other, Func &&func) const {
    for (const auto &[key, value] : map_) {
      auto iter = other.find(key);
      if (iter != other.end()) {
        if ((value - iter->second) > 0) {
          func(key);
        }
      } else {
        func(key);
      }
    }
  }

  std::vector<key_type> subtract(const counter_type &other) const {
    std::vector<key_type> result;
    subtract_by(other, [&result](const key_type &item) { result.emplace_back(item); });
    return result;
  }

  std::size_t size() const { return map_.size(); }

  bool contains(const key_type &key) const { return map_.contains(key); }

  const_iterator find(const key_type &key) const { return map_.find(key); }

  iterator begin() { return map_.begin(); }
  iterator end() { return map_.end(); }

  const_iterator begin() const { return map_.cbegin(); }
  const_iterator end() const { return map_.cend(); }

  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator cend() const { return map_.cend(); }

 private:
  map_type map_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_COUNTER_H_
