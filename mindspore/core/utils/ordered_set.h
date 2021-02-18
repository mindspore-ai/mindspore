/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_UTILS_ORDERED_SET_H_
#define MINDSPORE_CORE_UTILS_ORDERED_SET_H_

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <list>
#include <utility>
#include <string>
#include <functional>
#include <memory>
#include "utils/log_adapter.h"

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

  OrderedSet &operator=(const OrderedSet &os) {
    if (this != &os) {
      for (auto &item : os.ordered_data_) {
        add(item);
      }
    }
    return *this;
  }

  OrderedSet &operator=(OrderedSet &&os) = default;

  // insert an element to the OrderedSet after the given position.
  std::pair<iterator, bool> insert(iterator pos, const element_type &e) {
    auto result = mapped_data_.emplace(e, ordered_data_.end());
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
    auto pos = mapped_data_.find(e);
    if (pos == mapped_data_.end()) {
      return false;
    }
    // erase the sequential data first
    (void)ordered_data_.erase(pos->second);
    (void)mapped_data_.erase(pos);
    return true;
  }

  iterator erase(iterator pos) {
    (void)mapped_data_.erase(*pos);
    return ordered_data_.erase(pos);
  }

  iterator erase(const_iterator pos) {
    (void)mapped_data_.erase(*pos);
    return ordered_data_.erase(pos);
  }

  // Return the container size
  std::size_t size() const { return mapped_data_.size(); }

  bool empty() const { return mapped_data_.size() == 0; }

  // Return the string contents in orderset, using ordered_data
  std::string toString() {
    std::ostringstream res;
    res << "orderset content:\n";
    for (auto &item : ordered_data_) {
      res << std::to_string(reinterpret_cast<uintptr_t>(item.get())) << " ";
    }
    return res.str();
  }

  // Clear the elements
  void clear() {
    if (!mapped_data_.empty()) {
      mapped_data_.clear();
    }
    ordered_data_.clear();
  }

  // Compare two orderedset, if the order is not equal shall return false
  bool operator==(const OrderedSet &other) const { return ordered_data_ == other.ordered_data_; }

  // Remove and return the first element in the OrderedSet
  T pop() {
    if (ordered_data_.size() != 0) {
      T res = ordered_data_.front();
      (void)mapped_data_.erase(res);
      (void)ordered_data_.erase(ordered_data_.begin());
      return res;
    }
    MS_LOG(EXCEPTION) << "pop() on empty OrderedSet";
  }

  T &back() {
    if (ordered_data_.size() != 0) {
      return ordered_data_.back();
    }
    MS_LOG(EXCEPTION) << "back() on empty OrderedSet";
  }

  T &front() {
    if (ordered_data_.size() != 0) {
      return ordered_data_.front();
    }
    MS_LOG(EXCEPTION) << "front() on empty OrderedSet";
  }

  // Return true if there are no common elements
  bool is_disjoint(const OrderedSet &other) {
    for (auto &item : other.ordered_data_) {
      if (mapped_data_.find(item) != mapped_data_.end()) {
        return false;
      }
    }
    return true;
  }

  // Test whether this is subset of other
  bool is_subset(const OrderedSet &other) {
    for (auto &item : ordered_data_) {
      if (other.mapped_data_.find(item) == other.mapped_data_.end()) {
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
      if (other.mapped_data_.find(item) == other.mapped_data_.end()) {
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
      if (mapped_data_.find(item) != mapped_data_.end()) {
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

  bool contains(const element_type &e) const { return (mapped_data_.find(e) != mapped_data_.end()); }

  const_iterator find(const element_type &e) const {
    auto iter = mapped_data_.find(e);
    if (iter == mapped_data_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  iterator find(const element_type &e) {
    auto iter = mapped_data_.find(e);
    if (iter == mapped_data_.end()) {
      return ordered_data_.end();
    }
    return iter->second;
  }

  // Return the count of an element in set
  std::size_t count(const element_type &e) const { return mapped_data_.count(e); }

  iterator begin() { return ordered_data_.begin(); }
  iterator end() { return ordered_data_.end(); }

  const_iterator begin() const { return ordered_data_.cbegin(); }
  const_iterator end() const { return ordered_data_.cend(); }

  const_iterator cbegin() const { return ordered_data_.cbegin(); }
  const_iterator cend() const { return ordered_data_.cend(); }

 private:
  map_type mapped_data_;
  sequential_type ordered_data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_ORDERED_SET_H_
