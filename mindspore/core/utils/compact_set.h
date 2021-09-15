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

#ifndef MINDSPORE_CORE_UTILS_COMPACT_SET_H_
#define MINDSPORE_CORE_UTILS_COMPACT_SET_H_

#include <vector>
#include <utility>
#include <algorithm>

namespace mindspore {
// CompactSet uses a std::vector to hold data, it keeps insertion order
// but use less memory than OrderedSet. It could be more efficient than
// OrderedSet when used with a small number of elements.
template <typename T>
class CompactSet {
 public:
  using data_type = std::vector<T>;
  using iterator = typename data_type::iterator;
  using const_iterator = typename data_type::const_iterator;

  void add(T &&e) {
    auto iter = std::find(data_.begin(), data_.end(), e);
    if (iter == data_.end()) {
      data_.emplace_back(std::move(e));
    }
  }

  void insert(const T &e) {
    auto iter = std::find(data_.begin(), data_.end(), e);
    if (iter == data_.end()) {
      data_.push_back(e);
    }
  }

  iterator find(const T &e) { return std::find(data_.begin(), data_.end(), e); }

  const_iterator find(const T &e) const { return std::find(data_.begin(), data_.end(), e); }

  bool contains(const T &e) const { return (find(e) != data_.end()); }

  bool erase(const T &e) {
    auto iter = std::find(data_.begin(), data_.end(), e);
    if (iter == data_.end()) {
      return false;
    }
    data_.erase(iter);
    return true;
  }

  iterator erase(iterator pos) { return data_.erase(pos); }

  void clear() { data_.clear(); }

  const T &front() const { return data_.front(); }
  const T &back() const { return data_.back(); }

  T pop() {
    T e = std::move(data_.front());
    data_.erase(data_.begin());
    return e;
  }

  bool empty() const { return data_.empty(); }
  std::size_t size() const { return data_.size(); }

  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }

  const_iterator begin() const { return data_.cbegin(); }
  const_iterator end() const { return data_.cend(); }

  const_iterator cbegin() const { return data_.cbegin(); }
  const_iterator cend() const { return data_.cend(); }

 private:
  data_type data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_COMPACT_SET_H_
