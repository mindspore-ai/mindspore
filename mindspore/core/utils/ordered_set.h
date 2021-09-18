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

#ifndef MINDSPORE_CORE_UTILS_ORDERED_SET_H_
#define MINDSPORE_CORE_UTILS_ORDERED_SET_H_

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <list>
#include <utility>
#include <functional>
#include <memory>
#include "utils/hashing.h"

namespace mindspore {
// Implementation of OrderedSet that keeps insertion order
// using map as set, and use list as a sequential container to record elements to keep insertion order
template <class T, class Hash = std::hash<T>, class KeyEqual = std::equal_to<T>>
class OrderedSet {
 public:
  using element_type = T;
  using hasher = Hash;
  using equal = KeyEqual;
  using sequential_type = std::list<element_type>;
  using vector_type = std::vector<element_type>;
  using iterator = typename sequential_type::iterator;
  using const_iterator = typename sequential_type::const_iterator;
  using reverse_iterator = typename sequential_type::reverse_iterator;
  using const_reverse_iterator = typename sequential_type::const_reverse_iterator;
  using map_type = std::unordered_map<element_type, iterator, hasher, equal>;
  using ordered_set_type = OrderedSet<element_type, hasher, equal>;

  OrderedSet() = default;
  ~OrderedSet() = default;
  // OrderedSet use an iterator to list as mapped value to improve the performance of insertion and deletion,
  // So copy of OrderedSet should re-build value of the map key to make it pointer to the new list,, thus we use
  // traversal to build elements.
  OrderedSet(const OrderedSet &os) {
    for (auto &item : os.ordered_data_) {
      add(item);
    }
  }

  OrderedSet(OrderedSet &&os) = default;

  explicit OrderedSet(const sequential_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  // Explicitly construct an OrderedSet use vector
  explicit OrderedSet(const vector_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  OrderedSet &operator=(const OrderedSet &other) {
    if (this != &other) {
      clear();
      reserve(other.size());
      for (auto &item : other.ordered_data_) {
        add(item);
      }
    }
    return *this;
  }

  OrderedSet &operator=(OrderedSet &&other) = default;

  // insert an element to the OrderedSet after the given position.
  std::pair<iterator, bool> insert(iterator pos, const element_type &e) {
    auto result = map_.emplace(e, ordered_data_.end());
    if (result.second) {
      result.first->second = ordered_data_.emplace(pos, e);
    }
    return {result.first->second, result.second};
  }

  // Add an element to the OrderedSet, without judging return value
  void add(const element_type &e) { (void)insert(ordered_data_.end(), e); }

  // insert an element to the end of OrderedSet.
  std::pair<iterator, bool> insert(const element_type &e) { return insert(ordered_data_.end(), e); }

  void push_back(const element_type &e) { (void)insert(ordered_data_.end(), e); }

  void push_front(const element_type &e) { (void)insert(ordered_data_.begin(), e); }

  // Remove an element, if removed return true, otherwise return false
  bool erase(const element_type &e) {
    auto pos = map_.find(e);
    if (pos == map_.end()) {
      return false;
    }
    // erase the sequential data first
    (void)ordered_data_.erase(pos->second);
    (void)map_.erase(pos);
    return true;
  }

  iterator erase(iterator pos) {
    (void)map_.erase(*pos);
    return ordered_data_.erase(pos);
  }

  iterator erase(const_iterator pos) {
    (void)map_.erase(*pos);
    return ordered_data_.erase(pos);
  }

  // Return the container size
  std::size_t size() const { return map_.size(); }

  bool empty() const { return map_.size() == 0; }

  // Clear the elements
  void clear() {
    map_.clear();
    ordered_data_.clear();
  }

  // Reserve memory for the number of entries.
  void reserve(size_t num_entries) { map_.reserve(num_entries); }

  // Compare two orderedset, if the order is not equal shall return false
  bool operator==(const OrderedSet &other) const { return ordered_data_ == other.ordered_data_; }

  element_type pop() {
    element_type e = std::move(ordered_data_.front());
    (void)map_.erase(e);
    (void)ordered_data_.erase(ordered_data_.begin());
    return e;
  }

  element_type &back() { return ordered_data_.back(); }
  element_type &front() { return ordered_data_.front(); }

  const element_type &back() const { return ordered_data_.back(); }
  const element_type &front() const { return ordered_data_.front(); }

  // Return true if there are no common elements
  bool is_disjoint(const OrderedSet &other) {
    for (auto &item : other.ordered_data_) {
      if (map_.find(item) != map_.end()) {
        return false;
      }
    }
    return true;
  }

  // Test whether this is subset of other
  bool is_subset(const OrderedSet &other) {
    for (auto &item : ordered_data_) {
      if (other.map_.find(item) == other.map_.end()) {
        return false;
      }
    }
    return true;
  }

  // Add elements in other to this orderedset
  void update(const OrderedSet &other) {
    for (auto &item : other.ordered_data_) {
      add(item);
    }
  }

  void update(const std::shared_ptr<OrderedSet> &other) { update(*other); }

  void update(const sequential_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  void update(const vector_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  ordered_set_type get_union(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    res.update(other);
    return res;
  }

  // Get the union with other set, this operator may cost time because of copy
  ordered_set_type operator|(const OrderedSet &other) { return get_union(other); }

  // Return the intersection of two sets
  ordered_set_type intersection(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    for (auto &item : ordered_data_) {
      if (other.map_.find(item) == other.map_.end()) {
        (void)res.erase(item);
      }
    }
    return res;
  }
  ordered_set_type operator&(const OrderedSet &other) { return intersection(other); }

  // Return the symmetric difference of two sets
  ordered_set_type symmetric_difference(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    for (auto &item : other.ordered_data_) {
      if (map_.find(item) != map_.end()) {
        (void)res.erase(item);
      } else {
        res.add(item);
      }
    }
    return res;
  }

  ordered_set_type operator^(const OrderedSet &other) { return symmetric_difference(other); }

  // Remove elements which is also in others.
  void difference_update(const OrderedSet &other) {
    // use vector traversal, to keep ordrer
    for (auto &item : other.ordered_data_) {
      (void)erase(item);
    }
  }

  void difference_update(const sequential_type &other) {
    for (auto &item : other) {
      (void)erase(item);
    }
  }

  void difference_update(const vector_type &other) {
    for (auto &item : other) {
      (void)erase(item);
    }
  }

  // Return the set with elements that are not in the others
  ordered_set_type difference(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    res.difference_update(other);
    return res;
  }
  ordered_set_type operator-(const OrderedSet &other) { return difference(other); }

  bool contains(const element_type &e) const { return (map_.find(e) != map_.end()); }

  const_iterator find(const element_type &e) const {
    auto iter = map_.find(e);
    if (iter == map_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  iterator find(const element_type &e) {
    auto iter = map_.find(e);
    if (iter == map_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  // Return the count of an element in set
  std::size_t count(const element_type &e) const { return map_.count(e); }

  iterator begin() { return ordered_data_.begin(); }
  iterator end() { return ordered_data_.end(); }

  const_iterator begin() const { return ordered_data_.cbegin(); }
  const_iterator end() const { return ordered_data_.cend(); }

  const_iterator cbegin() const { return ordered_data_.cbegin(); }
  const_iterator cend() const { return ordered_data_.cend(); }

 private:
  map_type map_;
  sequential_type ordered_data_;
};

// OrderedSet that specially optimized for shared_ptr.
template <class T>
class OrderedSet<std::shared_ptr<T>> {
 public:
  using element_type = std::shared_ptr<T>;
  using key_type = const T *;
  using hash_t = PointerHash<T>;
  using sequential_type = std::list<element_type>;
  using vector_type = std::vector<element_type>;
  using iterator = typename sequential_type::iterator;
  using const_iterator = typename sequential_type::const_iterator;
  using reverse_iterator = typename sequential_type::reverse_iterator;
  using const_reverse_iterator = typename sequential_type::const_reverse_iterator;
  using map_type = std::unordered_map<key_type, iterator, hash_t>;
  using ordered_set_type = OrderedSet<std::shared_ptr<T>>;

  OrderedSet() = default;
  ~OrderedSet() = default;

  OrderedSet(const OrderedSet &os) {
    for (auto &item : os.ordered_data_) {
      add(item);
    }
  }

  OrderedSet(OrderedSet &&os) = default;

  explicit OrderedSet(const sequential_type &other) {
    reserve(other.size());
    for (auto &item : other) {
      add(item);
    }
  }

  explicit OrderedSet(const vector_type &other) {
    reserve(other.size());
    for (auto &item : other) {
      add(item);
    }
  }

  OrderedSet &operator=(const OrderedSet &other) {
    if (this != &other) {
      clear();
      reserve(other.size());
      for (auto &item : other.ordered_data_) {
        add(item);
      }
    }
    return *this;
  }

  OrderedSet &operator=(OrderedSet &&other) = default;

  std::pair<iterator, bool> insert(iterator pos, const element_type &e) {
    auto [map_iter, inserted] = map_.emplace(e.get(), iterator{});
    if (inserted) {
      map_iter->second = ordered_data_.emplace(pos, e);
    }
    return {map_iter->second, inserted};
  }

  std::pair<iterator, bool> insert(iterator pos, element_type &&e) {
    auto [map_iter, inserted] = map_.emplace(e.get(), iterator{});
    if (inserted) {
      map_iter->second = ordered_data_.emplace(pos, std::move(e));
    }
    return {map_iter->second, inserted};
  }

  void add(const element_type &e) { (void)insert(ordered_data_.end(), e); }

  void add(element_type &&e) { (void)insert(ordered_data_.end(), std::move(e)); }

  std::pair<iterator, bool> insert(const element_type &e) { return insert(ordered_data_.end(), e); }

  std::pair<iterator, bool> insert(element_type &&e) { return insert(ordered_data_.end(), std::move(e)); }

  void push_back(const element_type &e) { (void)insert(ordered_data_.end(), e); }

  void push_front(const element_type &e) { (void)insert(ordered_data_.begin(), e); }

  bool erase(const element_type &e) {
    auto pos = map_.find(e.get());
    if (pos == map_.end()) {
      return false;
    }
    auto iter = pos->second;
    (void)map_.erase(pos);
    (void)ordered_data_.erase(iter);
    return true;
  }

  iterator erase(iterator pos) {
    (void)map_.erase(pos->get());
    return ordered_data_.erase(pos);
  }

  iterator erase(const_iterator pos) {
    (void)map_.erase(pos->get());
    return ordered_data_.erase(pos);
  }

  std::size_t size() const { return ordered_data_.size(); }

  bool empty() const { return ordered_data_.empty(); }

  void clear() {
    map_.clear();
    ordered_data_.clear();
  }

  void reserve(size_t num_entries) { map_.reserve(num_entries); }

  bool operator==(const OrderedSet &other) const { return ordered_data_ == other.ordered_data_; }

  element_type pop() {
    element_type e = std::move(ordered_data_.front());
    (void)map_.erase(e.get());
    (void)ordered_data_.erase(ordered_data_.begin());
    return e;
  }

  element_type &back() { return ordered_data_.back(); }
  element_type &front() { return ordered_data_.front(); }

  const element_type &back() const { return ordered_data_.back(); }
  const element_type &front() const { return ordered_data_.front(); }

  // Return true if there are no common elements.
  bool is_disjoint(const OrderedSet &other) {
    return std::all_of(begin(), end(), [&other](const auto &e) { return !other.contains(e); });
  }

  // Test whether this is subset of other.
  bool is_subset(const OrderedSet &other) {
    return std::all_of(begin(), end(), [&other](const auto &e) { return other.contains(e); });
  }

  // Add elements in other to this orderedset.
  void update(const OrderedSet &other) {
    for (auto &item : other.ordered_data_) {
      add(item);
    }
  }

  void update(const std::shared_ptr<OrderedSet> &other) { update(*other); }

  void update(const sequential_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  void update(const vector_type &other) {
    for (auto &item : other) {
      add(item);
    }
  }

  ordered_set_type get_union(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    res.update(other);
    return res;
  }

  // Get the union with other set, this operator may cost time because of copy.
  ordered_set_type operator|(const OrderedSet &other) { return get_union(other); }

  // Return the intersection of two sets.
  ordered_set_type intersection(const OrderedSet &other) {
    ordered_set_type res;
    for (auto &item : ordered_data_) {
      if (other.contains(item)) {
        res.add(item);
      }
    }
    return res;
  }

  ordered_set_type operator&(const OrderedSet &other) { return intersection(other); }

  // Return the symmetric difference of two sets.
  ordered_set_type symmetric_difference(const OrderedSet &other) {
    ordered_set_type res(ordered_data_);
    for (auto &item : other) {
      if (contains(item)) {
        (void)res.erase(item);
      } else {
        res.add(item);
      }
    }
    return res;
  }

  ordered_set_type operator^(const OrderedSet &other) { return symmetric_difference(other); }

  // Remove elements which is also in others.
  void difference_update(const OrderedSet &other) {
    for (auto &item : other) {
      (void)erase(item);
    }
  }

  void difference_update(const sequential_type &other) {
    for (auto &item : other) {
      (void)erase(item);
    }
  }

  void difference_update(const vector_type &other) {
    for (auto &item : other) {
      (void)erase(item);
    }
  }

  // Return the set with elements that are not in the others.
  ordered_set_type difference(const OrderedSet &other) {
    ordered_set_type res;
    for (auto &item : ordered_data_) {
      if (!other.contains(item)) {
        res.add(item);
      }
    }
    return res;
  }

  ordered_set_type operator-(const OrderedSet &other) { return difference(other); }

  bool contains(const element_type &e) const { return (map_.find(e.get()) != map_.end()); }

  const_iterator find(const element_type &e) const {
    auto iter = map_.find(e.get());
    if (iter == map_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  iterator find(const element_type &e) {
    auto iter = map_.find(e.get());
    if (iter == map_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  std::size_t count(const element_type &e) const { return map_.count(e.get()); }

  iterator begin() { return ordered_data_.begin(); }
  iterator end() { return ordered_data_.end(); }

  const_iterator begin() const { return ordered_data_.cbegin(); }
  const_iterator end() const { return ordered_data_.cend(); }

  const_iterator cbegin() const { return ordered_data_.cbegin(); }
  const_iterator cend() const { return ordered_data_.cend(); }

 private:
  map_type map_;
  sequential_type ordered_data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_ORDERED_SET_H_
