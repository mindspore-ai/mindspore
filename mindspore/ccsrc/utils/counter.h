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

#ifndef MINDSPORE_CCSRC_UTILS_COUNTER_H_
#define MINDSPORE_CCSRC_UTILS_COUNTER_H_
#include <functional>
#include "utils/ordered_map.h"

namespace mindspore {

template <typename T, class Hash = std::hash<T>, class Equal = std::equal_to<T>>
class Counter {
  using counter_type = Counter<T, Hash, Equal>;

 public:
  Counter() = default;
  ~Counter() = default;

  Counter(const Counter &other) { data = other.data; }
  Counter &operator=(const Counter &other) {
    if (this != &other) {
      data = other.data;
    }
    return *this;
  }

  int &operator[](const T &t) { return data[t]; }

  counter_type operator-(const counter_type &other) {
    counter_type new_counter;
    for (auto iter = begin(); iter != end(); ++iter) {
      auto key = iter->first;
      int value = iter->second;
      auto item = other.data.find(key);
      if (item != other.data.end()) {
        int o_value = item->second;
        if (value - o_value > 0) {
          new_counter[key] = value - o_value;
        }
      } else {
        new_counter[key] = value;
      }
    }

    return new_counter;
  }

  counter_type operator+(const counter_type &other) {
    counter_type new_counter;
    for (auto iter = begin(); iter != end(); ++iter) {
      auto key = iter->first;
      int value = iter->second;
      auto item = other.data.find(key);
      if (item != other.data.end()) {
        new_counter[key] = iter->second + item->second;
      } else {
        new_counter[key] = value;
      }
    }

    for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
      auto key = iter->first;
      int value = iter->second;
      if (!new_counter.contains(key)) {
        new_counter[key] = value;
      }
    }

    return new_counter;
  }

  std::size_t size() const { return data.size(); }

  bool contains(const T &t) const { return data.find(t) != data.end(); }

  typename OrderedMap<T, int, Hash, Equal>::iterator begin() { return data.begin(); }

  typename OrderedMap<T, int, Hash, Equal>::iterator end() { return data.end(); }

  typename OrderedMap<T, int, Hash, Equal>::const_iterator cbegin() const { return data.cbegin(); }

  typename OrderedMap<T, int, Hash, Equal>::const_iterator cend() const { return data.cend(); }

 private:
  OrderedMap<T, int, Hash, Equal> data;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_COUNTER_H_
