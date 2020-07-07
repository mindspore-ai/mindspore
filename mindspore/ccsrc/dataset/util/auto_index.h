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
#ifndef DATASET_UTIL_AUTO_INDEX_H_
#define DATASET_UTIL_AUTO_INDEX_H_

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "dataset/util/btree.h"
#include "dataset/util/system_pool.h"

namespace mindspore {
namespace dataset {
/// This is a B+ tree with generated int64_t value as key.
/// Use minKey() function to query the min key.
/// Use maxKey() function to query the max key.
/// @tparam T
template <typename T, typename A = std::allocator<T>>
class AutoIndexObj : public BPlusTree<int64_t, T, A> {
 public:
  using my_tree = BPlusTree<int64_t, T, A>;
  using key_type = typename my_tree::key_type;
  using value_type = typename my_tree::value_type;

  AutoIndexObj() : my_tree::BPlusTree(), inx_(kMinKey) {}

  explicit AutoIndexObj(const Allocator<T> &alloc) : my_tree::BPlusTree(alloc), inx_(kMinKey) {}

  ~AutoIndexObj() = default;

  // Insert an object into the tree.
  // @param val
  // @return
  Status insert(const value_type &val, key_type *key = nullptr) {
    key_type my_inx = inx_.fetch_add(1);
    if (key != nullptr) {
      *key = my_inx;
    }
    return my_tree::DoInsert(my_inx, val);
  }

  Status insert(std::unique_ptr<value_type> &&val, key_type *key = nullptr) {
    key_type my_inx = inx_.fetch_add(1);
    if (key) {
      *key = my_inx;
    }
    return my_tree::DoInsert(my_inx, std::move(val));
  }

  // Insert a vector of objects into the tree.
  // @param v
  // @return
  Status insert(std::vector<value_type> v) {
    uint64_t num_ele = v.size();
    if (num_ele > 0) {
      // reserve a range of keys rather than getting it one by one.
      key_type my_inx = inx_.fetch_add(num_ele);
      for (uint64_t i = 0; i < num_ele; i++) {
        RETURN_IF_NOT_OK(my_tree::DoInsert(my_inx + i, v.at(i)));
      }
    }
    return Status::OK();
  }

  // @return the minimum key
  key_type min_key() const {
    auto it = this->cbegin();
    return it.key();
  }

  // @return the maximum key
  key_type max_key() const {
    auto it = this->cend();
    --it;
    return it.key();
  }

 private:
  static constexpr key_type kMinKey = 0;
  std::atomic<key_type> inx_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_AUTO_INDEX_H_
