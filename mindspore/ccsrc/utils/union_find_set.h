/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_UNION_FIND_SET_H_
#define MINDSPORE_CCSRC_UTILS_UNION_FIND_SET_H_

#include <map>
#include <set>

namespace mindspore {
template <class T>
class UnionFindSet {
 public:
  UnionFindSet() : union_find_set_() {}
  ~UnionFindSet() = default;
  void Add(const T &elem) {
    if (union_find_set_.find(elem) != union_find_set_.end()) {
      return;
    }

    union_find_set_[elem] = elem;
  }

  T Find(const T &key) {
    T key_parent = key;
    auto iter = union_find_set_.find(key_parent);
    if (iter == union_find_set_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "union_find_set_ cannot find key " << key_parent;
    }
    while (key_parent != iter->second) {
      key_parent = iter->second;
      iter = union_find_set_.find(key_parent);
      if (iter == union_find_set_.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "union_find_set_ cannot find key " << key_parent;
      }
    }

    T tmp = key;
    T tmp_parent;
    while (tmp != key_parent) {
      iter = union_find_set_.find(tmp);
      if (iter == union_find_set_.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "union_find_set_ cannot find key " << tmp;
      }
      tmp_parent = iter->second;
      union_find_set_[tmp] = key_parent;
      tmp = tmp_parent;
    }
    return key_parent;
  }

  void Union(const T &left, const T &right) { union_find_set_[Find(left)] = Find(right); }

  std::map<T, std::set<T>> GetSets() {
    std::map<T, std::set<T>> result;
    for (auto &iter : union_find_set_) {
      (void)Find(iter.first);
    }
    for (auto &iter : union_find_set_) {
      T parent = Find(iter.first);
      result[parent].insert(iter.first);
    }
    return result;
  }

 private:
  std::map<T, T> union_find_set_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_UNION_FIND_SET_H_
