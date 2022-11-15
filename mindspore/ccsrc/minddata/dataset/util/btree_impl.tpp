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
template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::InnerNode::Sort() {
  // Build an inverse map. Basically it means keys[i] should be relocated to keys[inverse[i]];
  slot_allocator alloc(this->alloc_);
  try {
    // We use a unique_ptr will custom deleter to ensure the memory will be released when this
    // function returns.
    std::unique_ptr<slot_type[], std::function<void(slot_type *)>> memGuard(
      alloc.allocate(traits::kInnerSlots), [&alloc](slot_type *p) { alloc.deallocate(p, traits::kInnerSlots); });
    slot_type *inverse = memGuard.get();
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
        if ((j + 1) >= traits::kInnerSlots + 1 || (i + 1) >= traits::kInnerSlots + 1) {
          return IndexRc::kUnexpectedError;
        }
        std::swap(data_[j + 1], data_[i + 1]);
        // one key in order.
        inverse[j] = j;
        // continue to move
        inverse[i] = k;
      }
      slot_dir_[i] = i;
    }
    return IndexRc::kOk;
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::InnerNode::Split(
  BPlusTree<K, V, A, C, T>::InnerNode *to, key_type *split_key) {
  MS_ASSERT(to);
  MS_ASSERT(to->slotuse_ == 0);
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
  err = memcpy_s(to->data_, sizeof(to->data_), data_ + mid + 1, (num_keys_to_move + 1) * sizeof(BaseNode *));
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

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::InnerNode::InsertIntoSlot(
  slot_type slot, const key_type &key, BPlusTree<K, V, A, C, T>::BaseNode *ptr) {
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
    if (err != EOK) {
      return IndexRc::kUnexpectedError;
    }
  }
  slot_dir_[slot] = slotuse_;
  keys_[slotuse_] = key;
  data_[slotuse_ + 1] = ptr;
  ++slotuse_;
  return IndexRc::kOk;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::LeafNode::Sort() {
  // Build an inverse map. Basically it means keys[i] should be relocated to keys[inverse[i]];
  slot_allocator alloc(this->alloc_);
  try {
    // We use a unique_ptr will custom deleter to ensure the memory will be released when this
    // function returns.
    std::unique_ptr<slot_type[], std::function<void(slot_type *)>> memGuard(
      alloc.allocate(traits::kLeafSlots), [&alloc](slot_type *p) { alloc.deallocate(p, traits::kLeafSlots); });
    slot_type *inverse = memGuard.get();
    for (slot_type i = 0; i < slotuse_; i++) {
      inverse[slot_dir_[i]] = i;
    }
    for (slot_type i = 0; i < slotuse_; i++) {
      while (inverse[i] != i) {
        slot_type j = inverse[i];
        slot_type k = inverse[j];
        // Swap the key
        if (j >= traits::kLeafSlots || i >= traits::kLeafSlots) {
          return IndexRc::kUnexpectedError;
        }
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
    return IndexRc::kOk;
  } catch (std::bad_alloc &e) {
    return IndexRc::kOutOfMemory;
  } catch (std::exception &e) {
    return IndexRc::kUnexpectedError;
  }
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::LeafNode::Split(
  BPlusTree<K, V, A, C, T>::LeafNode *to) {
  MS_ASSERT(to);
  MS_ASSERT(to->slotuse_ == 0);
  // It is simpler to sort first, then split. Other alternative is to move key by key to the
  // new node. Also we need to deal with the 'holes' after a key is moved.
  RETURN_IF_BAD_RC(this->Sort());
  slot_type mid = slotuse_ >> 1;
  slot_type num_keys_to_move = slotuse_ - mid;
  errno_t err = memmove_s(to->keys_, sizeof(to->keys_), keys_ + mid, num_keys_to_move * sizeof(key_type));
  if (err != EOK) {
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

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::LeafNode::InsertIntoSlot(
  BPlusTree<K, V, A, C, T>::LockPathCB *insCB, slot_type slot, const key_type &key,
  std::unique_ptr<value_type> &&value) {
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
    if (err != EOK) {
      return IndexRc::kUnexpectedError;
    }
  }
  slot_dir_[slot] = slotuse_;
  keys_[slotuse_] = key;
  data_[slotuse_] = std::move(value);
  ++slotuse_;
  return IndexRc::kOk;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::AllocateInner(
  BPlusTree<K, V, A, C, T>::InnerNode **p) {
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
  *p = new (ptr) InnerNode(alloc_);
  all_.Prepend(ptr);
  stats_.inner_nodes_++;
  return IndexRc::kOk;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::AllocateLeaf(
  BPlusTree<K, V, A, C, T>::LeafNode **p) {
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
  *p = new (ptr) LeafNode(alloc_);
  all_.Prepend(ptr);
  stats_.leaves_++;
  return IndexRc::kOk;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::LeafInsertKeyValue(
  BPlusTree<K, V, A, C, T>::LockPathCB *ins_cb, BPlusTree<K, V, A, C, T>::LeafNode *node, const key_type &key,
  std::unique_ptr<value_type> &&value, key_type *split_key, BPlusTree<K, V, A, C, T>::LeafNode **split_node) {
  bool duplicate;
  slot_type slot = FindSlot(node, key, &duplicate);
  if (duplicate) {
    return IndexRc::kDuplicateKey;
  }
  IndexRc rc = node->InsertIntoSlot(ins_cb, slot, key, std::move(value));
  if (rc == IndexRc::kSlotFull) {
    LeafNode *new_leaf = nullptr;
    rc = AllocateLeaf(&new_leaf);
    RETURN_IF_BAD_RC(rc);
    leaf_nodes_.InsertAfter(node, new_leaf);
    *split_node = new_leaf;
    // 50/50 split
    rc = node->Split(new_leaf);
    RETURN_IF_BAD_RC(rc);
    *split_key = new_leaf->keys_[0];
    if (LessThan(key, *split_key)) {
      rc = node->InsertIntoSlot(nullptr, slot, key, std::move(value));
      RETURN_IF_BAD_RC(rc);
    } else {
      slot -= node->slotuse_;
      rc = new_leaf->InsertIntoSlot(nullptr, slot, key, std::move(value));
      RETURN_IF_BAD_RC(rc);
    }
  }
  return rc;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::InnerInsertKeyChild(
  BPlusTree<K, V, A, C, T>::InnerNode *node, const key_type &key, BPlusTree<K, V, A, C, T>::BaseNode *ptr,
  key_type *split_key, BPlusTree<K, V, A, C, T>::InnerNode **split_node) {
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
  return rc;
}

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::InsertKeyValue(
  BPlusTree<K, V, A, C, T>::LockPathCB *ins_cb, BPlusTree<K, V, A, C, T>::BaseNode *n, const key_type &key,
  std::unique_ptr<value_type> &&value, key_type *split_key, BPlusTree<K, V, A, C, T>::BaseNode **split_node) {
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

template <typename K, typename V, typename A, typename C, typename T>
typename BPlusTree<K, V, A, C, T>::IndexRc BPlusTree<K, V, A, C, T>::Locate(RWLock *parent_lock, bool forUpdate,
                                                                            BPlusTree<K, V, A, C, T>::BaseNode *top,
                                                                            const key_type &key,
                                                                            BPlusTree<K, V, A, C, T>::LeafNode **ln,
                                                                            slot_type *s) const {
  if (ln == nullptr || s == nullptr) {
    return IndexRc::kNullPointer;
  }
  if (top == nullptr) {
    return IndexRc::kKeyNotFound;
  }
  RWLock *myLock = nullptr;
  if (parent_lock != nullptr) {
    // Crabbing. Lock this node first, then unlock the parent.
    myLock = &top->rw_lock_;
    if (top->is_leafnode()) {
      if (forUpdate) {
        // We are holding the parent lock in S and try to lock this node with X. It is not possible to run
        // into deadlock because no one will hold the child in X and trying to lock the parent in that order.
        myLock->LockExclusive();
      } else {
        myLock->LockShared();
      }
    } else {
      myLock->LockShared();
    }
    parent_lock->Unlock();
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
      if (myLock != nullptr) {
        myLock->Unlock();
      }
      return IndexRc::kKeyNotFound;
    }
  } else {
    auto *inner = static_cast<InnerNode *>(top);
    slot_type slot = FindSlot(inner, key);
    return Locate(myLock, forUpdate, FindBranch(inner, slot), key, ln, s);
  }
  // We still have a S lock on the leaf node. Leave it there. The iterator will unlock it for us.
  return IndexRc::kOk;
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::BPlusTree()
    : leaf_nodes_(&LeafNode::link_), all_(&BaseNode::lru_), root_(nullptr), acquire_lock_(true) {
  Init();
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::BPlusTree(const Allocator<V> &alloc)
    : alloc_(alloc), leaf_nodes_(&LeafNode::link_), all_(&BaseNode::lru_), root_(nullptr), acquire_lock_(true) {
  Init();
}

template <typename K, typename V, typename A, typename C, typename T>
BPlusTree<K, V, A, C, T>::~BPlusTree() noexcept {
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

template <typename K, typename V, typename A, typename C, typename T>
Status BPlusTree<K, V, A, C, T>::DoInsert(const key_type &key, std::unique_ptr<value_type> &&value) {
  IndexRc rc;
  bool retry = false;
  do {
    // Track all the paths to the target and lock each internal node in S.
    LockPathCB InsCB(this, retry);
    // Initially we lock path in S unless we need to do node split.
    retry = false;
    BaseNode *new_child = nullptr;
    key_type new_key = key_type();
    rc = InsertKeyValue(acquire_lock_ ? &InsCB : nullptr, root_, key, std::move(value), &new_key, &new_child);
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
  (void)stats_.size_++;
  return Status::OK();
}

template <typename K, typename V, typename A, typename C, typename T>
Status BPlusTree<K, V, A, C, T>::DoInsert(const key_type &key, const value_type &value) {
  // We don't store the value directly into the leaf node as it is expensive to move it during node split.
  // Rather we store a pointer instead.
  return DoInsert(key, std::make_unique<value_type>(value));
}

template <typename K, typename V, typename A, typename C, typename T>
std::unique_ptr<V> BPlusTree<K, V, A, C, T>::DoUpdate(const key_type &key, const value_type &new_value) {
  return DoUpdate(key, std::make_unique<value_type>(new_value));
}

template <typename K, typename V, typename A, typename C, typename T>
std::unique_ptr<V> BPlusTree<K, V, A, C, T>::DoUpdate(const key_type &key, std::unique_ptr<value_type> &&new_value) {
  if (root_ != nullptr) {
    LeafNode *leaf = nullptr;
    slot_type slot;
    RWLock *myLock = nullptr;
    if (acquire_lock_) {
      myLock = &this->rw_lock_;
      // Lock the tree in S, pass the lock to Locate which will unlock it for us underneath.
      myLock->LockShared();
    }
    IndexRc rc = Locate(myLock, true, root_, key, &leaf, &slot);
    if (rc == IndexRc::kOk) {
      // All locks from the tree to the parent of leaf are all gone. We still have a X lock
      // on the leaf.
      // Swap out the old value and replace it with new value.
      std::unique_ptr<value_type> old = std::move(leaf->data_[leaf->slot_dir_[slot]]);
      leaf->data_[leaf->slot_dir_[slot]] = std::move(new_value);
      if (acquire_lock_) {
        leaf->rw_lock_.Unlock();
      }
      return old;
    } else {
      MS_LOG(DEBUG) << "Key not found. rc = " << static_cast<int>(rc) << ".";
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

}  // namespace dataset
}  // namespace mindspore
#endif
