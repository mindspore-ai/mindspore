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

#ifndef MINDSPORE_CORE_UTILS_ORDERED_MAP_H_
#define MINDSPORE_CORE_UTILS_ORDERED_MAP_H_

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <list>
#include <string>
#include <functional>
#include <memory>
#include "utils/log_adapter.h"

namespace mindspore {
// Implementation of OrderedMap that keeps insertion order
// using unordered_map to improve the performance of find/erase, and use list to keep insertion order
template <typename KeyT, typename ValueT, class Hash = std::hash<KeyT>, class Equal = std::equal_to<KeyT>>
class OrderedMap {
 public:
  using key_t = KeyT;
  using value_t = ValueT;
  using hasher = Hash;
  using equal = Equal;
  using pair_type = std::pair<key_t, value_t>;
  using sequential_type = std::list<pair_type>;
  using iterator = typename sequential_type::iterator;
  using const_iterator = typename sequential_type::const_iterator;
  using reverse_iterator = typename sequential_type::reverse_iterator;
  using const_reverse_iterator = typename sequential_type::const_reverse_iterator;
  using map_type = std::unordered_map<key_t, iterator, hasher, equal>;
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
  OrderedMap(const OrderedMap &os) {
    for (auto &item : os.sequential_data_) {
      (void)insert(pair_type(item.first, item.second));
    }
  }

  // Explicitly construct OrderedMap use sequential_type
  explicit OrderedMap(const sequential_type &other) {
    for (auto &item : other) {
      (void)insert(pair_type(item.first, item.second));
    }
  }

  OrderedMap &operator=(const OrderedMap &os) {
    if (this != &os) {
      for (auto &item : os.sequential_data_) {
        (void)insert(pair_type(item.first, item.second));
      }
    }
    return *this;
  }

  void clear() {
    if (!map_data_.empty()) {
      map_data_.clear();
    }
    sequential_data_.clear();
  }

  void swap(OrderedMap &rhs) {
    std::swap(map_data_, rhs.map_data_);
    std::swap(sequential_data_, rhs.sequential_data_);
  }

  void reserve(size_type num_entries) {
    map_data_.reserve(num_entries);
    sequential_data_.reserve(num_entries);
  }

  std::pair<iterator, bool> add(const key_t &key) {
    iterator empty_itr;
    std::pair<key_t, typename map_type::mapped_type> map_pair = std::make_pair(key, empty_itr);
    std::pair<typename map_type::iterator, bool> result = map_data_.insert(map_pair);
    auto &seq_itr = result.first->second;
    if (result.second) {
      auto it = sequential_data_.insert(sequential_data_.end(), std::make_pair(key, ValueT()));
      seq_itr = it;
    }
    return std::pair<iterator, bool>(seq_itr, result.second);
  }

  ValueT &operator[](const key_t &key) {
    auto result = add(key);
    return (*result.first).second;
  }

  std::pair<iterator, bool> insert(const pair_type &kv) {
    auto result = add(kv.first);
    if (result.second) {
      *(result.first) = kv;
      return std::make_pair(std::prev(end()), true);
    }
    return std::make_pair(result.first, false);
  }

  std::pair<iterator, bool> insert(pair_type &&kv) {
    iterator empty_itr;
    std::pair<key_t, typename map_type::mapped_type> map_pair = std::make_pair(kv.first, empty_itr);
    std::pair<typename map_type::iterator, bool> result = map_data_.insert(map_pair);
    auto &seq_itr = result.first->second;
    if (result.second) {
      auto it = sequential_data_.insert(sequential_data_.end(), std::move(kv));
      seq_itr = it;
      return std::make_pair(std::prev(end()), true);
    }
    return std::make_pair(seq_itr, false);
  }

  bool empty() const { return sequential_data_.empty(); }

  size_type size() const { return sequential_data_.size(); }

  size_type count(const key_t &key) const {
    auto pos = map_data_.find(key);
    return pos == map_data_.end() ? 0 : 1;
  }

  iterator find(const key_t &key) {
    typename map_type::const_iterator pos = map_data_.find(key);
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  const_iterator find(const key_t &key) const {
    auto pos = map_data_.find(key);
    return pos == map_data_.end() ? sequential_data_.end() : (pos->second);
  }

  ValueT at(const key_t &key) {
    auto pos = map_data_.find(key);
    if (pos == map_data_.end()) {
      MS_LOG(EXCEPTION) << "Have no key " << key;
    }
    return pos->second->second;
  }

  // Remove the last element from the sequential_data_.
  void pop_back() {
    typename map_type::iterator pos = map_data_.find(sequential_data_.back().first);
    map_data_.erase(pos);
    sequential_data_.pop_back();
  }

  // Remove the first element from the sequential_data_.
  void pop_front() {
    typename map_type::iterator pos = map_data_.find(sequential_data_.first().first);
    map_data_.erase(pos);
    sequential_data_.pop_front();
  }

  // Remove the element given by Iterator.
  typename sequential_type::iterator erase(const typename sequential_type::iterator &itr) {
    (void)map_data_.erase(itr->first);
    auto next = sequential_data_.erase(itr);
    if (next == sequential_data_.end()) return next;
    return next;
  }

  // Remove the element with the given key
  size_type erase(const key_t &key) {
    auto itr = find(key);
    if (itr == end()) return 0;
    (void)erase(itr);
    return 1;
  }

  void update(const key_t &old_key, const key_t &new_key) {
    auto old_it = find(old_key);
    if (old_it == end()) {
      return;
    }
    auto new_it = find(new_key);
    if (new_it == end()) {
      old_it->first = new_key;
      auto nh = map_data_.extract(old_key);
      nh.key() = new_key;
      map_data_.insert(std::move(nh));
      return;
    }
    *old_it = *new_it;
    (void)erase(old_key);
    (void)erase(new_key);
  }

 private:
  map_type map_data_;
  sequential_type sequential_data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_ORDERED_MAP_H_
