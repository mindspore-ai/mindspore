/* COPYRIGHT 2019 Huawei Technologies Co., Ltd.All Rights Reserved.
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
#ifndef DATASET_UTIL_BTREE_ITERATOR_H_
#define DATASET_UTIL_BTREE_ITERATOR_H_

#include "dataset/util/de_error.h"
#include "utils/log_adapter.h"
#include "btree.h"

namespace mindspore {
namespace dataset {
template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator &BPlusTree<K, V, C, T>::Iterator::operator++() {
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return *this;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator BPlusTree<K, V, C, T>::Iterator::operator++(int) {
  Iterator tmp = *this;
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return tmp;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator &BPlusTree<K, V, C, T>::Iterator::operator--() {
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return *this;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator BPlusTree<K, V, C, T>::Iterator::operator--(int) {
  Iterator tmp = *this;
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return tmp;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator &BPlusTree<K, V, C, T>::ConstIterator::operator++() {
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return *this;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::ConstIterator::operator++(int) {
  Iterator tmp = *this;
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return tmp;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator &BPlusTree<K, V, C, T>::ConstIterator::operator--() {
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return *this;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::ConstIterator::operator--(int) {
  Iterator tmp = *this;
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return tmp;
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::Search(const key_type &key) const {
  if (root_ != nullptr) {
    LeafNode *leaf = nullptr;
    slot_type slot;
    IndexRc rc = Locate(root_, key, &leaf, &slot);
    if (rc == IndexRc::kOk) {
      return ConstIterator(leaf, slot);
    } else {
      MS_LOG(INFO) << "Key not found. rc = " << static_cast<int>(rc) << ".";
      return end();
    }
  } else {
    return end();
  }
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::value_type BPlusTree<K, V, C, T>::operator[](key_type key) {
  ConstIterator it = Search(key);
  return it.value();
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator BPlusTree<K, V, C, T>::begin() {
  return Iterator(this);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::Iterator BPlusTree<K, V, C, T>::end() {
  return Iterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::begin() const {
  return ConstIterator(this);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::end() const {
  return ConstIterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::cbegin() const {
  return ConstIterator(this);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::ConstIterator BPlusTree<K, V, C, T>::cend() const {
  return ConstIterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}
}  // namespace dataset
}  // namespace mindspore
#endif
