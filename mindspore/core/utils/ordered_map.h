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

#ifndef MINDSPORE_CORE_UTILS_ORDERED_MAP_H_
#define MINDSPORE_CORE_UTILS_ORDERED_MAP_H_

#include <list>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <utility>
#include <memory>
#include "utils/hashing.h"

namespace mindspore {
// Implementation of OrderedMap that keeps insertion order
// using unordered_map to improve the performance of find/erase, and use list to keep insertion order
template <typename KeyT, typename ValueT, class Hash = std::hash<KeyT>, class Equal = std::equal_to<KeyT>>
class OrderedMap {
  using key_ptr_t = const KeyT *;
  struct KeyPtrHash {
    std::size_t operator()(key_ptr_t ptr) const noexcept { return Hash{}(*ptr); }
  };
  struct KeyPtrEqual {
    bool operator()(key_ptr_t lhs, key_ptr_t rhs) const noexcept { return Equal{}(*lhs, *rhs); }
  };

 public:
  using key_t = KeyT;
  using value_t = ValueT;
  using pair_type = std::pair<key_t, value_t>;
  using sequential_type = std::list<pair_type>;
  using iterator = typename sequential_type::iterator;
  using const_iterator = typename sequential_type::const_iterator;
  using reverse_iterator = typename sequential_type::reverse_iterator;
  using const_reverse_iterator = typename sequential_type::const_reverse_iterator;
  using map_type = std::unordered_map<key_ptr_t, iterator, KeyPtrHash, KeyPtrEqual>;
  using value_type = typename sequential_type::value_type;
  using size_type = typename sequential_type::size_type;

  iterator begin() { return sequential_data_.begin(); }
  iterator end() { return sequential_data_.end(); }
  const_iterator begin() const { return sequential_data_.cbegin(); }
  const_iterator end() const { return sequential_data_.cend(); }
  const_iterator cbegin() const { return sequential_data_.cbegin(); }
  const_iterator cend() const { return sequential_data_.cend(); }

  reverse_iterator rbegin() { return sequential_data_.rbegin(); }
  reverse_iterator rend() { return sequential_data_.rend(); }
  const_reverse_iterator rbegin() const { return sequential_data_.rbegin(); }
  const_reverse_iterator rend() const { return sequential_data_.rend(); }

  pair_type &front() { return sequential_data_.front(); }
  const pair_type &front() const { return sequential_data_.front(); }
  pair_type &back() { return sequential_data_.back(); }
  const pair_type &back() const { return sequential_data_.back(); }

  OrderedMap() = default;
  ~OrderedMap() = default;

  OrderedMap(OrderedMap &&other) noexcept = default;
  OrderedMap &operator=(OrderedMap &&other) noexcept = default;

  explicit OrderedMap(const sequential_type &other) {
    reserve(other.size());
    for (auto &item : other) {
      (void)emplace(item.first, item.second);
    }
  }

  OrderedMap(const OrderedMap &other) : OrderedMap(other.sequential_data_) {}

  OrderedMap &operator=(const OrderedMap &other) {
    if (this != &other) {
      clear();
      reserve(other.size());
      for (auto &item : other.sequential_data_) {
        (void)emplace(item.first, item.second);
      }
    }
    return *this;
  }

  void clear() {
    map_data_.clear();
    sequential_data_.clear();
  }

  void swap(OrderedMap &rhs) noexcept {
    std::swap(map_data_, rhs.map_data_);
    std::swap(sequential_data_, rhs.sequential_data_);
  }

  void reserve(size_type num_entries) { map_data_.reserve(num_entries); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&... args) {
    auto new_iter = sequential_data_.emplace(sequential_data_.end(), std::forward<Args>(args)...);
    auto [map_iter, inserted] = map_data_.emplace(&(new_iter->first), new_iter);
    if (!inserted) {
      sequential_data_.erase(new_iter);
    }
    return {map_iter->second, inserted};
  }

  std::pair<iterator, bool> insert(const pair_type &kv) {
    auto iter = map_data_.find(&(kv.first));
    if (iter != map_data_.end()) {
      return {iter->second, false};
    }
    auto new_iter = sequential_data_.emplace(sequential_data_.end(), kv);
    auto result = map_data_.emplace(&(new_iter->first), new_iter);
    return {result.first->second, true};
  }

  std::pair<iterator, bool> insert(pair_type &&kv) {
    auto iter = map_data_.find(&(kv.first));
    if (iter != map_data_.end()) {
      return {iter->second, false};
    }
    auto new_iter = sequential_data_.emplace(sequential_data_.end(), std::move(kv));
    auto result = map_data_.emplace(&(new_iter->first), new_iter);
    return {result.first->second, true};
  }

  std::pair<iterator, bool> add(const key_t &key) { return insert(pair_type{key, ValueT{}}); }

  ValueT &operator[](const key_t &key) {
    auto iter = map_data_.find(&key);
    if (iter != map_data_.end()) {
      return iter->second->second;
    }
    auto new_iter = sequential_data_.emplace(sequential_data_.end(), key, ValueT{});
    auto result = map_data_.emplace(&(new_iter->first), new_iter);
    return result.first->second->second;
  }

  bool empty() const { return sequential_data_.empty(); }

  size_type size() const { return sequential_data_.size(); }

  const ValueT &at(const key_t &key) const {
    auto &list_iter = map_data_.at(&key);
    return list_iter->second;
  }

  size_type count(const key_t &key) const {
    auto pos = map_data_.find(&key);
    return pos == map_data_.end() ? 0 : 1;
  }

  iterator find(const key_t &key) {
    auto pos = map_data_.find(&key);
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  const_iterator find(const key_t &key) const {
    auto pos = map_data_.find(&key);
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  // Remove the last element from the sequential_data_.
  void pop_back() {
    (void)map_data_.erase(&(sequential_data_.back().first));
    sequential_data_.pop_back();
  }

  // Remove the first element from the sequential_data_.
  void pop_front() {
    (void)map_data_.erase(&(sequential_data_.front().first));
    sequential_data_.pop_front();
  }

  // Remove the element given by Iterator.
  iterator erase(iterator iter) {
    (void)map_data_.erase(&(iter->first));
    return sequential_data_.erase(iter);
  }

  // Remove the element with the given key
  size_type erase(const key_t &key) {
    auto itr = find(key);
    if (itr == end()) {
      return 0;
    }
    (void)erase(itr);
    return 1;
  }

 private:
  map_type map_data_;
  sequential_type sequential_data_;
};

// OrderedMap that specially optimized for shared_ptr key type.
template <typename T, typename ValueT>
class OrderedMap<std::shared_ptr<T>, ValueT> {
 public:
  using raw_key_t = const T *;
  using key_t = std::shared_ptr<T>;
  using hash_t = PointerHash<T>;
  using value_t = ValueT;
  using pair_type = std::pair<key_t, value_t>;
  using sequential_type = std::list<pair_type>;
  using iterator = typename sequential_type::iterator;
  using const_iterator = typename sequential_type::const_iterator;
  using reverse_iterator = typename sequential_type::reverse_iterator;
  using const_reverse_iterator = typename sequential_type::const_reverse_iterator;
  using map_type = std::unordered_map<raw_key_t, iterator, hash_t>;
  using value_type = typename sequential_type::value_type;
  using size_type = typename sequential_type::size_type;

  iterator begin() { return sequential_data_.begin(); }
  iterator end() { return sequential_data_.end(); }
  const_iterator begin() const { return sequential_data_.cbegin(); }
  const_iterator end() const { return sequential_data_.cend(); }
  const_iterator cbegin() const { return sequential_data_.cbegin(); }
  const_iterator cend() const { return sequential_data_.cend(); }

  reverse_iterator rbegin() { return sequential_data_.rbegin(); }
  reverse_iterator rend() { return sequential_data_.rend(); }
  const_reverse_iterator rbegin() const { return sequential_data_.rbegin(); }
  const_reverse_iterator rend() const { return sequential_data_.rend(); }

  pair_type &front() { return sequential_data_.front(); }
  const pair_type &front() const { return sequential_data_.front(); }
  pair_type &back() { return sequential_data_.back(); }
  const pair_type &back() const { return sequential_data_.back(); }

  OrderedMap() = default;
  ~OrderedMap() = default;

  OrderedMap(OrderedMap &&other) noexcept = default;
  OrderedMap &operator=(OrderedMap &&other) noexcept = default;

  explicit OrderedMap(const sequential_type &other) {
    reserve(other.size());
    for (auto &item : other) {
      (void)emplace(item.first, item.second);
    }
  }

  OrderedMap(const OrderedMap &other) : OrderedMap(other.sequential_data_) {}

  OrderedMap &operator=(const OrderedMap &other) {
    if (this != &other) {
      clear();
      reserve(other.size());
      for (auto &item : other.sequential_data_) {
        (void)emplace(item.first, item.second);
      }
    }
    return *this;
  }

  void clear() {
    map_data_.clear();
    sequential_data_.clear();
  }

  void swap(OrderedMap &rhs) noexcept {
    std::swap(map_data_, rhs.map_data_);
    std::swap(sequential_data_, rhs.sequential_data_);
  }

  void reserve(size_type num_entries) { map_data_.reserve(num_entries); }

  template <typename K, typename V>
  std::pair<iterator, bool> emplace(K &&key, V &&value) {
    auto [map_iter, inserted] = map_data_.emplace(key.get(), iterator{});
    if (inserted) {
      map_iter->second = sequential_data_.emplace(sequential_data_.end(), std::forward<K>(key), std::forward<V>(value));
    }
    return {map_iter->second, inserted};
  }

  std::pair<iterator, bool> insert(const pair_type &kv) {
    auto [map_iter, inserted] = map_data_.emplace(kv.first.get(), iterator{});
    if (inserted) {
      map_iter->second = sequential_data_.emplace(sequential_data_.end(), kv);
    }
    return {map_iter->second, inserted};
  }

  std::pair<iterator, bool> insert(pair_type &&kv) {
    auto [map_iter, inserted] = map_data_.emplace(kv.first.get(), iterator{});
    if (inserted) {
      map_iter->second = sequential_data_.emplace(sequential_data_.end(), std::move(kv));
    }
    return {map_iter->second, inserted};
  }

  std::pair<iterator, bool> add(const key_t &key) { return insert(pair_type{key, ValueT{}}); }

  ValueT &operator[](const key_t &key) {
    auto result = emplace(key, ValueT{});
    return result.first->second;
  }

  bool empty() const { return sequential_data_.empty(); }

  size_type size() const { return sequential_data_.size(); }

  const ValueT &at(const key_t &key) const {
    auto &list_iter = map_data_.at(key.get());
    return list_iter->second;
  }

  size_type count(const key_t &key) const {
    auto pos = map_data_.find(key.get());
    return pos == map_data_.end() ? 0 : 1;
  }

  iterator find(const key_t &key) {
    auto pos = map_data_.find(key.get());
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  const_iterator find(const key_t &key) const {
    auto pos = map_data_.find(key.get());
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  // Remove the last element from the sequential_data_.
  void pop_back() {
    (void)map_data_.erase(sequential_data_.back().first.get());
    sequential_data_.pop_back();
  }

  // Remove the first element from the sequential_data_.
  void pop_front() {
    (void)map_data_.erase(sequential_data_.front().first.get());
    sequential_data_.pop_front();
  }

  // Remove the element given by Iterator.
  iterator erase(iterator iter) {
    (void)map_data_.erase(iter->first.get());
    return sequential_data_.erase(iter);
  }

  // Remove the element with the given key.
  size_type erase(const key_t &key) {
    auto itr = find(key);
    if (itr == end()) {
      return 0;
    }
    (void)erase(itr);
    return 1;
  }

 private:
  map_type map_data_;
  sequential_type sequential_data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_ORDERED_MAP_H_
