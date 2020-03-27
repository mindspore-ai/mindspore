/* Copyright 2019 Huawei Technologies Co., Ltd.All Rights Reserved.
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
#ifndef DATASET_UTIL_BTREE_H_
#define DATASET_UTIL_BTREE_H_

#include "btree.h"

namespace mindspore {
namespace dataset {
template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::InnerNode::Sort() {
  // Build an inverse map. Basically it means keys[i] should be relocated to keys[inverse[i]];
  Allocator<slot_type> alloc(this->alloc_);
  slot_type *inverse = nullptr;
  try {
    inverse = alloc.allocate(traits::kInnerSlots);
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }

  for (slot_type i = 0; i < slotuse_; i++) {
    inverse[slot_dir_[i]] = i;
  }
  for (slot_type i = 0; i < slotuse_; i++) {
    while (inverse[i] != i) {
      slot_type j = inverse[i];
      slot_type k = inverse[j];
      // Swap the key
      std::swap(keys_[j], keys_[i]);
      // Swap the pointers.
      std::swap(data_[j + 1], data_[i + 1]);
      // one key in order.
      inverse[j] = j;
      // continue to move
      inverse[i] = k;
    }
    slot_dir_[i] = i;
  }
  if (inverse != nullptr) {
    alloc.deallocate(inverse);
    inverse = nullptr;
  }
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::InnerNode::Split(BPlusTree<K, V, C, T>::InnerNode *to,
                                                                                key_type *split_key) {
  DS_ASSERT(to);
  DS_ASSERT(to->slotuse_ == 0);
  // It is simpler to sort first, then split. Other alternative is to move key by key to the
  // new node. Also we need to deal with the 'holes' after a key is moved.
  RETURN_IF_BAD_RC(this->Sort());
  slot_type mid = slotuse_ >> 1;
  slot_type num_keys_to_move = slotuse_ - (mid + 1);
  *split_key = keys_[mid];
  errno_t err = memmove_s(to->keys_, sizeof(to->keys_), keys_ + mid + 1, num_keys_to_move * sizeof(key_type));
  if (err != EOK) {
    return IndexRc::kUnexpectedError;
  }
  err = memcpy_s(to->data_, sizeof(to->data_), data_ + mid + 1, (num_keys_to_move + 1) * sizeof(BaseNode * ));
  if (err != EOK) {
    return IndexRc::kUnexpectedError;
  }
  for (slot_type i = 0; i < num_keys_to_move; i++) {
    to->slot_dir_[i] = i;
  }
  slotuse_ -= (num_keys_to_move + 1);  // the split key is moved up. So one less
  to->slotuse_ += num_keys_to_move;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::InnerNode::InsertIntoSlot(slot_type slot, const key_type &key,
                                                 BPlusTree<K, V, C, T>::BaseNode *ptr) {
  if (is_full()) {
    return IndexRc::kSlotFull;
  }
  // Shift the slot entries to the right and make room for the new comer.
  // We don't sort the key and/or the data array until node split
  auto num_keys_to_move = slotuse_ - slot;
  if (num_keys_to_move > 0) {
    auto *src = &slot_dir_[slot];
    auto *dest = &slot_dir_[slot + 1];
    auto destMax = sizeof(slot_dir_) - sizeof(slot_type) * (slot + 1);
    auto amt = sizeof(slot_type) * num_keys_to_move;
    errno_t err = memmove_s(dest, destMax, src, amt);
    if (err) {
      return IndexRc::kUnexpectedError;
    }
  }
  slot_dir_[slot] = slotuse_;
  keys_[slotuse_] = key;
  data_[slotuse_ + 1] = ptr;
  ++slotuse_;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::LeafNode::Sort() {
  // Build an inverse map. Basically it means keys[i] should be relocated to keys[inverse[i]];
  Allocator<slot_type> alloc(this->alloc_);
  slot_type *inverse = nullptr;
  try {
    inverse = alloc.allocate(traits::kLeafSlots);
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }

  for (slot_type i = 0; i < slotuse_; i++) {
    inverse[slot_dir_[i]] = i;
  }
  for (slot_type i = 0; i < slotuse_; i++) {
    while (inverse[i] != i) {
      slot_type j = inverse[i];
      slot_type k = inverse[j];
      // Swap the key
      std::swap(keys_[j], keys_[i]);
      // Swap the shared pointers
      std::swap(data_[j], data_[i]);
      // one key in order.
      inverse[j] = j;
      // continue to move
      inverse[i] = k;
    }
    slot_dir_[i] = i;
  }
  if (inverse != nullptr) {
    alloc.deallocate(inverse);
    inverse = nullptr;
  }
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::LeafNode::Split(BPlusTree<K, V, C, T>::LeafNode *to) {
  DS_ASSERT(to);
  DS_ASSERT(to->slotuse_ == 0);
  // It is simpler to sort first, then split. Other alternative is to move key by key to the
  // new node. Also we need to deal with the 'holes' after a key is moved.
  RETURN_IF_BAD_RC(this->Sort());
  slot_type mid = slotuse_ >> 1;
  slot_type num_keys_to_move = slotuse_ - mid;
  errno_t err = memmove_s(to->keys_, sizeof(to->keys_), keys_ + mid, num_keys_to_move * sizeof(key_type));
  if (err) {
    return IndexRc::kUnexpectedError;
  }
  for (slot_type i = 0; i < num_keys_to_move; i++) {
    to->data_[i] = std::move(data_[i + mid]);
    to->slot_dir_[i] = i;
  }
  slotuse_ -= num_keys_to_move;
  to->slotuse_ += num_keys_to_move;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::LeafNode::InsertIntoSlot(BPlusTree<K, V, C, T>::LockPathCB *insCB, slot_type slot,
                                                const key_type &key,
                                                std::shared_ptr<value_type> value) {
  if (is_full()) {
    // If we need to do node split, we need to ensure all the intermediate nodes are locked exclusive.
    // Otherwise we need to do a retry.
    if (insCB == nullptr || !insCB->latch_shared_) {
      return IndexRc::kSlotFull;
    } else {
      return IndexRc::kRetry;
    }
  }
  // We can now let go all the locks of the parent. Nothing we do from now on will change the
  // structure of the tree.
  if (insCB) {
    insCB->UnlockMyParents(this);
  }
  // Shift the slot entries to the right and make room for the new comer.
  // We don't sort the key and/or the data array until node split
  auto num_keys_to_move = slotuse_ - slot;
  if (num_keys_to_move > 0) {
    auto *src = &slot_dir_[slot];
    auto *dest = &slot_dir_[slot + 1];
    auto destMax = sizeof(slot_dir_) - sizeof(slot_type) * (slot + 1);
    auto amt = sizeof(slot_type) * num_keys_to_move;
    errno_t err = memmove_s(dest, destMax, src, amt);
    if (err) {
      return IndexRc::kUnexpectedError;
    }
  }
  slot_dir_[slot] = slotuse_;
  keys_[slotuse_] = key;
  data_[slotuse_] = std::move(value);
  ++slotuse_;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::AllocateInner(BPlusTree<K, V, C, T>::InnerNode **p) {
  if (p == nullptr) {
    return IndexRc::kNullPointer;
  }
  typename InnerNode::alloc_type alloc(alloc_);
  InnerNode *ptr = nullptr;
  try {
    ptr = alloc.allocate(1);
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }
  *p = new(ptr) InnerNode(alloc_);
  all_.Prepend(ptr);
  stats_.inner_nodes_++;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc BPlusTree<K, V, C, T>::AllocateLeaf(BPlusTree<K, V, C, T>::LeafNode **p) {
  if (p == nullptr) {
    return IndexRc::kNullPointer;
  }
  typename LeafNode::alloc_type alloc(this->alloc_);
  LeafNode *ptr = nullptr;
  try {
    ptr = alloc.allocate(1);
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }
  *p = new(ptr) LeafNode(alloc_);
  all_.Prepend(ptr);
  stats_.leaves_++;
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::LeafInsertKeyValue(BPlusTree<K, V, C, T>::LockPathCB *ins_cb,
                                          BPlusTree<K, V, C, T>::LeafNode *node, const key_type &key,
                                          std::shared_ptr<value_type> value, key_type *split_key,
                                          BPlusTree<K, V, C, T>::LeafNode **split_node) {
  bool duplicate;
  slot_type slot = FindSlot(node, key, &duplicate);
  if (duplicate) {
    return IndexRc::kDuplicateKey;
  }
  IndexRc rc = node->InsertIntoSlot(ins_cb, slot, key, value);
  if (rc == IndexRc::kSlotFull) {
    LeafNode *new_leaf = nullptr;
    rc = AllocateLeaf(&new_leaf);
    RETURN_IF_BAD_RC(rc);
    leaf_nodes_.InsertAfter(node, new_leaf);
    *split_node = new_leaf;
    if (slot == node->slotuse_ && traits::kAppendMode) {
      // Split high. Good for bulk load and keys are in asending order on insert
      *split_key = key;
      // Just insert the new key to the new leaf. No further need to move the keys
      // from one leaf to the other.
      rc = new_leaf->InsertIntoSlot(nullptr, 0, key, value);
      RETURN_IF_BAD_RC(rc);
    } else {
      // 50/50 split
      rc = node->Split(new_leaf);
      RETURN_IF_BAD_RC(rc);
      *split_key = new_leaf->keys_[0];
      if (LessThan(key, *split_key)) {
        rc = node->InsertIntoSlot(nullptr, slot, key, value);
        RETURN_IF_BAD_RC(rc);
      } else {
        slot -= node->slotuse_;
        rc = new_leaf->InsertIntoSlot(nullptr, slot, key, value);
        RETURN_IF_BAD_RC(rc);
      }
    }
  }
  return rc;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::InnerInsertKeyChild(BPlusTree<K, V, C, T>::InnerNode *node, const key_type &key,
                                           BPlusTree<K, V, C, T>::BaseNode *ptr,
                                           key_type *split_key, BPlusTree<K, V, C, T>::InnerNode **split_node) {
  bool duplicate;
  slot_type slot = FindSlot(node, key, &duplicate);
  if (duplicate) {
    return IndexRc::kDuplicateKey;
  }
  IndexRc rc = node->InsertIntoSlot(slot, key, ptr);
  if (rc == IndexRc::kSlotFull) {
    InnerNode *new_inner = nullptr;
    rc = AllocateInner(&new_inner);
    RETURN_IF_BAD_RC(rc);
    *split_node = new_inner;
    if (slot == node->slotuse_ && traits::kAppendMode) {
      *split_key = key;
      new_inner->data_[0] = node->data_[node->slotuse_];
      rc = new_inner->InsertIntoSlot(0, key, ptr);
      RETURN_IF_BAD_RC(rc);
    } else {
      rc = node->Split(new_inner, split_key);
      RETURN_IF_BAD_RC(rc);
      if (LessThan(key, *split_key)) {
        // Need to readjust the slot position since the split key is no longer in the two children.
        slot = FindSlot(node, key);
        rc = node->InsertIntoSlot(slot, key, ptr);
        RETURN_IF_BAD_RC(rc);
      } else {
        // Same reasoning as above
        slot = FindSlot(new_inner, key);
        rc = new_inner->InsertIntoSlot(slot, key, ptr);
        RETURN_IF_BAD_RC(rc);
      }
    }
  }
  return rc;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::InsertKeyValue(BPlusTree<K, V, C, T>::LockPathCB *ins_cb, BPlusTree<K, V, C, T>::BaseNode *n,
                                      const key_type &key,
                                      std::shared_ptr<value_type> value, key_type *split_key,
                                      BPlusTree<K, V, C, T>::BaseNode **split_node) {
  if (split_key == nullptr || split_node == nullptr) {
    return IndexRc::kUnexpectedError;
  }
  if (n->is_leafnode()) {
    if (ins_cb) {
      // Always lock the leaf in X.
      ins_cb->LockNode(n, LockPathCB::LockMode::kExclusive);
    }
    auto *leaf = static_cast<LeafNode *>(n);
    LeafNode *new_leaf = nullptr;
    RETURN_IF_BAD_RC(LeafInsertKeyValue(ins_cb, leaf, key, std::move(value), split_key, &new_leaf));
    if (new_leaf) {
      *split_node = new_leaf;
    }
  } else {
    if (ins_cb) {
      // For internal node, lock in S unless we are doing retry.
      if (ins_cb->latch_shared_) {
        ins_cb->LockNode(n, LockPathCB::LockMode::kShared);
      } else {
        ins_cb->LockNode(n, LockPathCB::LockMode::kExclusive);
      }
    }
    auto *inner = static_cast<InnerNode *>(n);
    slot_type slot = FindSlot(inner, key);
    BaseNode *new_child = nullptr;
    key_type new_key = key_type();
    RETURN_IF_BAD_RC(InsertKeyValue(ins_cb, FindBranch(inner, slot), key, std::move(value), &new_key, &new_child));
    if (new_child) {
      InnerNode *new_inner = nullptr;
      RETURN_IF_BAD_RC(InnerInsertKeyChild(inner, new_key, new_child, split_key, &new_inner));
      if (new_inner) {
        *split_node = new_inner;
      }
    }
  }
  return IndexRc::kOk;
}

template<typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::IndexRc
BPlusTree<K, V, C, T>::Locate(BPlusTree<K, V, C, T>::BaseNode *top, const key_type &key,
                              BPlusTree<K, V, C, T>::LeafNode **ln,
                              slot_type *s) const {
  if (ln == nullptr || s == nullptr) {
    return IndexRc::kNullPointer;
  }
  if (top == nullptr) {
    return IndexRc::kKeyNotFound;
  }
  if (top->is_leafnode()) {
    bool duplicate;
    auto *leaf = static_cast<LeafNode *>(top);
    slot_type slot = FindSlot(leaf, key, &duplicate);
    // Need exact match.
    if (duplicate) {
      *ln = leaf;
      *s = slot;
    } else {
      return IndexRc::kKeyNotFound;
    }
  } else {
    auto *inner = static_cast<InnerNode *>(top);
    slot_type slot = FindSlot(inner, key);
    return Locate(FindBranch(inner, slot), key, ln, s);
  }
  return IndexRc::kOk;
}

template <typename K, typename V, typename C, typename T>
BPlusTree<K, V, C, T>::BPlusTree(const value_allocator &alloc)
    : alloc_(alloc), leaf_nodes_(&LeafNode::link_), all_(&BaseNode::lru_), root_(nullptr) {}

template<typename K, typename V, typename C, typename T>
BPlusTree<K, V, C, T>::~BPlusTree() noexcept {
  // We have a list of all the nodes allocated. Traverse them and free all the memory
  BaseNode *n = all_.head;
  BaseNode *t = nullptr;
  while (n) {
    t = n->lru_.next;
    all_.Remove(n);
    if (n->is_leafnode()) {
      auto *leaf = static_cast<LeafNode *>(n);
      typename LeafNode::alloc_type alloc(alloc_);
      leaf->~LeafNode();
      alloc.deallocate(leaf, 1);
    } else {
      auto *in = static_cast<InnerNode *>(n);
      typename InnerNode::alloc_type alloc(alloc_);
      in->~InnerNode();
      alloc.deallocate(in, 1);
    }
    n = t;
  }
  root_ = nullptr;
}

template<typename K, typename V, typename C, typename T>
Status BPlusTree<K, V, C, T>::DoInsert(const key_type &key, const value_type &value) {
  IndexRc rc;
  if (root_ == nullptr) {
    UniqueLock lck(&rw_lock_);
    // Check again after we get the lock. Other thread may have created the root node already.
    if (root_ == nullptr) {
      LeafNode *leaf = nullptr;
      rc = AllocateLeaf(&leaf);
      if (rc != IndexRc::kOk) {
        return IndexRc2Status(rc);
      }
      leaf_nodes_.Append(leaf);
      root_ = leaf;
    }
    // lock will be unlocked when it goes out of scope.
  }
  bool retry = false;
  do {
    // Track all the paths to the target and lock each internal node in S.
    LockPathCB InsCB(this, retry);
    // Mark the numKeysArray invalid. We may latch the tree in S and multiple guys are doing insert.
    // But it is okay as we all set the same value.
    stats_.num_keys_array_valid_ = false;
    // Initially we lock path in S unless we need to do node split.
    retry = false;
    BaseNode *new_child = nullptr;
    key_type new_key = key_type();
    // We don't store the value directly into the leaf node as it is expensive to move it during node split.
    // Rather we store a pointer instead. The value_type must support the copy constructor.
    std::shared_ptr<value_type> ptr_value = std::make_shared<value_type>(value);
    rc = InsertKeyValue(&InsCB, root_, key, std::move(ptr_value), &new_key, &new_child);
    if (rc == IndexRc::kRetry) {
      retry = true;
    } else if (rc != IndexRc::kOk) {
      return IndexRc2Status(rc);
    } else if (new_child != nullptr) {
      // root is full
      InnerNode *new_root = nullptr;
      rc = AllocateInner(&new_root);
      if (rc == IndexRc::kOk) {
        rc = new_root->InsertIntoSlot(0, new_key, new_child);
        if (rc != IndexRc::kOk) {
          return IndexRc2Status(rc);
        }
        new_root->data_[0] = root_;
        root_ = new_root;
        stats_.level_++;
      } else {
        return IndexRc2Status(rc);
      }
    }
  } while (retry);
  (void) stats_.size_++;
  return Status::OK();
}

template <typename K, typename V, typename C, typename T>
void BPlusTree<K, V, C, T>::PopulateNumKeys() {
  // Start from the root and we calculate how many leaf nodes as pointed to by each inner node.
  // The results are stored in the numKeys array in each inner node.
  (void)PopulateNumKeys(root_);
  // Indicate the result is accurate since we have the tree locked exclusive.
  stats_.num_keys_array_valid_ = true;
}

template <typename K, typename V, typename C, typename T>
uint64_t BPlusTree<K, V, C, T>::PopulateNumKeys(BPlusTree<K, V, C, T>::BaseNode *n) {
  if (n->is_leafnode()) {
    auto *leaf = static_cast<LeafNode *>(n);
    return leaf->slotuse_;
  } else {
    auto *inner = static_cast<InnerNode *>(n);
    uint64_t num_keys = 0;
    for (auto i = 0; i < inner->slotuse_ + 1; i++) {
      inner->num_keys_[i] = PopulateNumKeys(inner->data_[i]);
      num_keys += inner->num_keys_[i];
    }
    return num_keys;
  }
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::key_type BPlusTree<K, V, C, T>::KeyAtPos(uint64_t inx) {
  if (stats_.num_keys_array_valid_ == false) {
    // We need exclusive access to the tree. If concurrent insert is going on, it is hard to get accurate numbers
    UniqueLock lck(&rw_lock_);
    // Check again.
    if (stats_.num_keys_array_valid_ == false) {
      PopulateNumKeys();
    }
  }
  // Now we know how many keys each inner branch contains, we can now traverse the correct node in log n time.
  return KeyAtPos(root_, inx);
}

template <typename K, typename V, typename C, typename T>
typename BPlusTree<K, V, C, T>::key_type BPlusTree<K, V, C, T>::KeyAtPos(BPlusTree<K, V, C, T>::BaseNode *n, uint64_t inx) {
  if (n->is_leafnode()) {
    auto *leaf = static_cast<LeafNode *>(n);
    return leaf->keys_[leaf->slot_dir_[inx]];
  } else {
    auto *inner = static_cast<InnerNode *>(n);
    if ((inx + 1) > inner->num_keys_[0]) {
      inx -= inner->num_keys_[0];
    } else {
      return KeyAtPos(inner->data_[0], inx);
    }
    for (auto i = 0; i < inner->slotuse_; i++) {
      if ((inx + 1) > inner->num_keys_[inner->slot_dir_[i] + 1]) {
        inx -= inner->num_keys_[inner->slot_dir_[i]+1];
      } else {
        return KeyAtPos(inner->data_[inner->slot_dir_[i] + 1], inx);
      }
    }
  }
  // If we get here, inx is way too big. Instead of throwing exception, we will just return the default value
  // of key_type whatever it is.
  return key_type();
}
}  // namespace dataset
}  // namespace mindspore
#endif
