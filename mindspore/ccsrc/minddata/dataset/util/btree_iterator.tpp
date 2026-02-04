/* COPYRIGHT 2019-2025 Huawei Technologies Co., Ltd.All Rights Reserved.
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

#include "include/securec.h"
#include "btree.h"

namespace mindspore::dataset {
template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::Iterator::~Iterator() {
  if (locked_) {
    cur_->rw_lock_.Unlock();
    locked_ = false;
  }
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator &BPlusTree<K, V, A, C, T>::Iterator::operator++() {
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    if (locked_) {
      cur_->link_.next->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator BPlusTree<K, V, A, C, T>::Iterator::operator++(int) {
  Iterator tmp = *this;
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    if (locked_) {
      cur_->link_.next->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return tmp;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator &BPlusTree<K, V, A, C, T>::Iterator::operator--() {
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    if (locked_) {
      cur_->link_.prev->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator BPlusTree<K, V, A, C, T>::Iterator::operator--(int) {
  Iterator tmp = *this;
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    if (locked_) {
      cur_->link_.prev->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return tmp;
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::Iterator::Iterator(const BPlusTree<K, V, A, C, T>::Iterator &lhs) {
  this->cur_ = lhs.cur_;
  this->slot_ = lhs.slot_;
  this->locked_ = lhs.locked_;
  if (this->locked_) {
    this->cur_->rw_lock_.LockShared();
  }
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::Iterator::Iterator(BPlusTree<K, V, A, C, T>::Iterator &&lhs) noexcept {
  this->cur_ = lhs.cur_;
  this->slot_ = lhs.slot_;
  this->locked_ = lhs.locked_;
  lhs.locked_ = false;
  lhs.slot_ = 0;
  lhs.cur_ = nullptr;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator &BPlusTree<K, V, A, C, T>::Iterator::operator=(
  const BPlusTree<K, V, A, C, T>::Iterator &lhs) {
  if (*this != lhs) {
    if (this->locked_) {
      this->cur_->rw_lock_.Unlock();
    }
    this->cur_ = lhs.cur_;
    this->slot_ = lhs.slot_;
    this->locked_ = lhs.locked_;
    if (this->locked_) {
      this->cur_->rw_lock_.LockShared();
    }
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator &BPlusTree<K, V, A, C, T>::Iterator::operator=(
  BPlusTree<K, V, A, C, T>::Iterator &&lhs) {
  if (*this != lhs) {
    if (this->locked_) {
      this->cur_->rw_lock_.Unlock();
    }
    this->cur_ = lhs.cur_;
    this->slot_ = lhs.slot_;
    this->locked_ = lhs.locked_;
    lhs.locked_ = false;
    lhs.slot_ = 0;
    lhs.cur_ = nullptr;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::ConstIterator::~ConstIterator() {
  if (locked_) {
    cur_->rw_lock_.Unlock();
    locked_ = false;
  }
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator &BPlusTree<K, V, A, C, T>::ConstIterator::operator++() {
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    if (locked_) {
      cur_->link_.next->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::ConstIterator::operator++(int) {
  Iterator tmp = *this;
  if (slot_ + 1u < cur_->slotuse_) {
    ++slot_;
  } else if (cur_->link_.next) {
    if (locked_) {
      cur_->link_.next->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.next;
    slot_ = 0;
  } else {
    slot_ = cur_->slotuse_;
  }
  return tmp;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator &BPlusTree<K, V, A, C, T>::ConstIterator::operator--() {
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    if (locked_) {
      cur_->link_.prev->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::ConstIterator::operator--(int) {
  Iterator tmp = *this;
  if (slot_ > 0) {
    --slot_;
  } else if (cur_->link_.prev) {
    if (locked_) {
      cur_->link_.prev->rw_lock_.LockShared();
      cur_->rw_lock_.Unlock();
    }
    cur_ = cur_->link_.prev;
    slot_ = cur_->slotuse_ - 1;
  } else {
    slot_ = 0;
  }
  return tmp;
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::ConstIterator::ConstIterator(const BPlusTree<K, V, A, C, T>::ConstIterator &lhs) {
  this->cur_ = lhs.cur_;
  this->slot_ = lhs.slot_;
  this->locked_ = lhs.locked_;
  if (this->locked_) {
    this->cur_->rw_lock_.LockShared();
  }
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::ConstIterator::ConstIterator(BPlusTree<K, V, A, C, T>::ConstIterator &&lhs) noexcept {
  this->cur_ = lhs.cur_;
  this->slot_ = lhs.slot_;
  this->locked_ = lhs.locked_;
  lhs.locked_ = false;
  lhs.slot_ = 0;
  lhs.cur_ = nullptr;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator &BPlusTree<K, V, A, C, T>::ConstIterator::operator=(
  const BPlusTree<K, V, A, C, T>::ConstIterator &lhs) {
  if (*this != lhs) {
    if (this->locked_) {
      this->cur_->rw_lock_.Unlock();
    }
    this->cur_ = lhs.cur_;
    this->slot_ = lhs.slot_;
    this->locked_ = lhs.locked_;
    if (this->locked_) {
      this->cur_->rw_lock_.LockShared();
    }
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator &BPlusTree<K, V, A, C, T>::ConstIterator::operator=(
  BPlusTree<K, V, A, C, T>::ConstIterator &&lhs) {
  if (*this != lhs) {
    if (this->locked_) {
      this->cur_->rw_lock_.Unlock();
    }
    this->cur_ = lhs.cur_;
    this->slot_ = lhs.slot_;
    this->locked_ = lhs.locked_;
    lhs.locked_ = false;
    lhs.slot_ = 0;
    lhs.cur_ = nullptr;
  }
  return *this;
}

template <typename K, typename V, typename A, typename C, typename T>
std::pair<typename BPlusTree<K, V, A, C, T>::ConstIterator, bool> BPlusTree<K, V, A, C, T>::Search(
  const key_type &key) const {
  if (root_ != nullptr) {
    LeafNode *leaf = nullptr;
    slot_type slot = slot_type();
    RWLock *myLock = nullptr;
    if (acquire_lock_) {
      myLock = &this->rw_lock_;
      // Lock the tree in S, pass the lock to Locate which will unlock it for us underneath.
      myLock->LockShared();
    }
    IndexRc rc = Locate(myLock, false, root_, key, &leaf, &slot);
    bool find = (rc == IndexRc::kOk);
    return std::make_pair(ConstIterator(leaf, slot, find), find);
  } else {
    return std::make_pair(cend(), false);
  }
}

template <typename K, typename V, typename A, typename C, typename T>
std::pair<typename BPlusTree<K, V, A, C, T>::Iterator, bool> BPlusTree<K, V, A, C, T>::Search(const key_type &key) {
  if (root_ != nullptr) {
    LeafNode *leaf = nullptr;
    slot_type slot = slot_type();
    RWLock *myLock = nullptr;
    if (acquire_lock_) {
      myLock = &this->rw_lock_;
      // Lock the tree in S, pass the lock to Locate which will unlock it for us underneath.
      myLock->LockShared();
    }
    IndexRc rc = Locate(myLock, false, root_, key, &leaf, &slot);
    bool find = (rc == IndexRc::kOk);
    return std::make_pair(Iterator(leaf, slot, find), find);
  } else {
    return std::make_pair(end(), false);
  }
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::value_type BPlusTree<K, V, A, C, T>::operator[](key_type key) {
  auto r = Search(key);
  return r.first.value();
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator BPlusTree<K, V, A, C, T>::begin() {
  return Iterator(this);
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::Iterator BPlusTree<K, V, A, C, T>::end() {
  return Iterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::begin() const {
  return ConstIterator(this);
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::end() const {
  return ConstIterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::cbegin() const {
  return ConstIterator(this);
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::ConstIterator BPlusTree<K, V, A, C, T>::cend() const {
  return ConstIterator(this->leaf_nodes_.tail, this->leaf_nodes_.tail ? this->leaf_nodes_.tail->slotuse_ : 0);
}
}  // namespace mindspore::dataset
#endif
