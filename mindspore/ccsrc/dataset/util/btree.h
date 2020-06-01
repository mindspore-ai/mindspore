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
#ifndef DATASET_UTIL_INDEX_H_
#define DATASET_UTIL_INDEX_H_

#include <algorithm>
#include <atomic>
#include <functional>
#include <utility>
#include <memory>
#include <deque>
#include "./securec.h"
#include "dataset/util/allocator.h"
#include "dataset/util/list.h"
#include "dataset/util/lock.h"
#include "dataset/util/memory_pool.h"
#include "dataset/util/services.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Default traits for a B+ tree
struct BPlusTreeTraits {
  // This determines the limit of number of keys in a node.
  using slot_type = uint16_t;
  // Number of slots in each leaf of the tree.
  static constexpr slot_type kLeafSlots = 256;
  // Number of slots in each inner node of the tree
  static constexpr slot_type kInnerSlots = 128;
  // If kAppendMode is true, we will split high instead of 50/50 split
  static constexpr bool kAppendMode = false;
};

/// Implementation of B+ tree
/// @tparam K -- the type of key
/// @tparam V -- the type of value
/// @tparam A -- allocator
/// @tparam C -- comparison class
/// @tparam T -- trait
template <typename K, typename V, typename A = std::allocator<V>, typename C = std::less<K>,
          typename T = BPlusTreeTraits>
class BPlusTree {
 public:
  enum class IndexRc : char {
    kOk = 0,
    kDuplicateKey = 1,
    kSlotFull = 2,
    kKeyNotFound = 3,
    kNullPointer = 4,
    kOutOfMemory = 5,
    kRetry = 6,
    kUnexpectedError = 127
  };
#define RETURN_IF_BAD_RC(_s)    \
  do {                          \
    IndexRc __rc = (_s);        \
    if (__rc != IndexRc::kOk) { \
      return __rc;              \
    }                           \
  } while (false)

  Status IndexRc2Status(IndexRc rc) {
    if (rc == IndexRc::kOk) {
      return Status(StatusCode::kOK);
    } else if (rc == IndexRc::kOutOfMemory) {
      return Status(StatusCode::kOutOfMemory);
    } else if (rc == IndexRc::kDuplicateKey) {
      return Status(StatusCode::kDuplicateKey);
    } else {
      RETURN_STATUS_UNEXPECTED(std::to_string(static_cast<int>(rc)));
    }
  }

  using key_type = K;
  using value_type = V;
  using key_compare = C;
  using slot_type = typename T::slot_type;
  using traits = T;
  using value_allocator = A;
  using key_allocator = typename value_allocator::template rebind<key_type>::other;
  using slot_allocator = typename value_allocator::template rebind<slot_type>::other;

  BPlusTree();

  explicit BPlusTree(const Allocator<V> &alloc);

  ~BPlusTree() noexcept;

  BPlusTree(const BPlusTree &) = delete;

  BPlusTree(BPlusTree &&) = delete;

  BPlusTree &operator=(const BPlusTree &) = delete;

  BPlusTree &operator=(BPlusTree &&) = delete;

  key_compare key_comp() const { return key_less_; }

  size_t size() const { return stats_.size_; }

  bool empty() const { return (size() == 0); }

  /// @param key
  /// @param value
  /// @return
  Status DoInsert(const key_type &key, const value_type &value);
  Status DoInsert(const key_type &key, std::unique_ptr<value_type> &&value);

  // Update a new value for a given key.
  std::unique_ptr<value_type> DoUpdate(const key_type &key, const value_type &new_value);
  std::unique_ptr<value_type> DoUpdate(const key_type &key, std::unique_ptr<value_type> &&new_value);

  void PopulateNumKeys();

  key_type KeyAtPos(uint64_t inx);

  // Statistics
  struct tree_stats {
    std::atomic<uint64_t> size_;
    uint32_t leaves_;
    uint32_t inner_nodes_;
    uint32_t level_;
    bool num_keys_array_valid_;

    tree_stats() : size_(0), leaves_(0), inner_nodes_(0), level_(0), num_keys_array_valid_(false) {}
  };

 private:
  // Abstract class of a node (leaf or inner)
  class BaseNode {
   public:
    friend class BPlusTree;

    virtual bool is_leafnode() const = 0;

    virtual bool is_full() const = 0;

    explicit BaseNode(const value_allocator &alloc) : alloc_(alloc) {}

    virtual ~BaseNode() = default;

   protected:
    mutable RWLock rw_lock_;
    value_allocator alloc_;

   private:
    Node<BaseNode> lru_;
  };

  uint64_t PopulateNumKeys(BaseNode *n);

  key_type KeyAtPos(BaseNode *n, uint64_t inx);

  // This control block keeps track of all the nodes we traverse on insert.
  // To maximize concurrency, internal nodes are latched S. If a node split
  // is required, we must releases all the latches and redo it again and change
  // the latch mode from S to X.
  struct LockPathCB {
    enum class LockMode : char { kShared = 0, kExclusive = 1, kNone = 2 };

    struct path {
      BaseNode *node_;
      bool locked_;

      path() : node_(nullptr), locked_(false) {}

      path(BaseNode *p, LockMode lockmode) : node_(p), locked_(false) {
        if (lockmode == LockMode::kExclusive) {
          p->rw_lock_.LockExclusive();
          locked_ = true;
        } else if (lockmode == LockMode::kShared) {
          p->rw_lock_.LockShared();
          locked_ = true;
        }
      }
    };

    LockPathCB(BPlusTree *tree, bool retryWithXlock) : self_(tree), latch_shared_(true) {
      if (retryWithXlock) {
        latch_shared_ = false;
      }
      if (latch_shared_) {
        tree->rw_lock_.LockShared();
      } else {
        tree->rw_lock_.LockExclusive();
      }
    }

    ~LockPathCB() noexcept {
      // Make sure all locks are released.
      while (!paths_.empty()) {
        path p = paths_.back();
        paths_.pop_back();
        if (p.locked_) {
          p.node_->rw_lock_.Unlock();
        }
      }
      self_->rw_lock_.Unlock();
      self_ = nullptr;
    }

    void LockNode(BaseNode *p, LockMode locktype) { paths_.emplace_back(p, locktype); }

    void UnlockMyParents(BaseNode *me) {
      path p = paths_.front();
      while (p.node_ != me) {
        if (p.locked_) {
          p.node_->rw_lock_.Unlock();
        }
        paths_.pop_front();
        p = paths_.front();
      }
    }

    BPlusTree *self_;
    std::deque<path> paths_;
    bool latch_shared_;
  };

  // Definition of inner node which fans to either inner node or leaf node.
  class InnerNode : public BaseNode {
   public:
    friend class BPlusTree;

    using alloc_type = typename value_allocator::template rebind<InnerNode>::other;

    bool is_leafnode() const override { return false; }

    bool is_full() const override { return (slotuse_ == traits::kInnerSlots); }

    IndexRc Sort();

    // 50/50 split
    IndexRc Split(InnerNode *to, key_type *split_key);

    IndexRc InsertIntoSlot(slot_type slot, const key_type &key, BaseNode *ptr);

    explicit InnerNode(const value_allocator &alloc) : BaseNode::BaseNode(alloc), slotuse_(0) {}

    ~InnerNode() = default;

    slot_type slot_dir_[traits::kInnerSlots] = {0};
    key_type keys_[traits::kInnerSlots] = {0};
    BaseNode *data_[traits::kInnerSlots + 1] = {nullptr};
    uint64_t num_keys_[traits::kInnerSlots + 1] = {0};
    slot_type slotuse_;
  };

  // Definition of a leaf node which contains the key/value pair
  class LeafNode : public BaseNode {
   public:
    friend class BPlusTree;

    using alloc_type = typename value_allocator::template rebind<LeafNode>::other;
    Node<LeafNode> link_;

    bool is_leafnode() const override { return true; }

    bool is_full() const override { return (slotuse_ == traits::kLeafSlots); }

    IndexRc Sort();

    // 50/50 split
    IndexRc Split(LeafNode *to);

    IndexRc InsertIntoSlot(LockPathCB *insCB, slot_type slot, const key_type &key, std::unique_ptr<value_type> &&value);

    explicit LeafNode(const value_allocator &alloc) : BaseNode::BaseNode(alloc), slotuse_(0) {}

    ~LeafNode() = default;

    slot_type slot_dir_[traits::kLeafSlots] = {0};
    key_type keys_[traits::kLeafSlots] = {0};
    std::unique_ptr<value_type> data_[traits::kLeafSlots];
    slot_type slotuse_;
  };

  mutable RWLock rw_lock_;
  value_allocator alloc_;
  // All the leaf nodes. Used by the iterator to traverse all the key/values.
  List<LeafNode> leaf_nodes_;
  // All the nodes (inner + leaf). Used by the destructor to free the memory of all the nodes.
  List<BaseNode> all_;
  // Pointer to the root of the tree.
  BaseNode *root_;
  // Key comparison object
  key_compare key_less_;
  // Stat
  tree_stats stats_;

  bool LessThan(const key_type &a, const key_type &b) const { return key_less_(a, b); }

  bool EqualOrLessThan(const key_type &a, const key_type &b) const { return !key_less_(b, a); }

  bool Equal(const key_type &a, const key_type &b) const { return !key_less_(a, b) && !key_less_(b, a); }

  IndexRc AllocateInner(InnerNode **p);

  IndexRc AllocateLeaf(LeafNode **p);

  template <typename node_type>
  slot_type FindSlot(const node_type *node, const key_type &key, bool *duplicate = nullptr) const {
    slot_type lo = 0;
    while (lo < node->slotuse_ && key_comp()(node->keys_[node->slot_dir_[lo]], key)) {
      ++lo;
    }
    bool keymatch = (lo < node->slotuse_ && Equal(key, node->keys_[node->slot_dir_[lo]]));
    if (keymatch && !node->is_leafnode()) {
      // For an inner node and we match a key during search, we should look into the next slot.
      ++lo;
    }
    if (duplicate != nullptr) {
      *duplicate = keymatch;
    }
    return lo;
  }

  IndexRc LeafInsertKeyValue(LockPathCB *ins_cb, LeafNode *node, const key_type &key,
                             std::unique_ptr<value_type> &&value, key_type *split_key, LeafNode **split_node);

  IndexRc InnerInsertKeyChild(InnerNode *node, const key_type &key, BaseNode *ptr, key_type *split_key,
                              InnerNode **split_node);

  inline BaseNode *FindBranch(InnerNode *inner, slot_type slot) const {
    BaseNode *child = nullptr;
    if (slot == 0) {
      child = inner->data_[0];
    } else {
      child = inner->data_[inner->slot_dir_[slot - 1] + 1];
    }
    return child;
  }

  IndexRc InsertKeyValue(LockPathCB *ins_cb, BaseNode *n, const key_type &key, std::unique_ptr<value_type> &&value,
                         key_type *split_key, BaseNode **split_node);

  IndexRc Locate(RWLock *parent_lock, bool forUpdate, BaseNode *top, const key_type &key, LeafNode **ln,
                 slot_type *s) const;

 public:
  class Iterator : public std::iterator<std::bidirectional_iterator_tag, value_type> {
   public:
    using reference = BPlusTree::value_type &;
    using pointer = BPlusTree::value_type *;

    explicit Iterator(BPlusTree *btree) : cur_(btree->leaf_nodes_.head), slot_(0), locked_(false) {}

    Iterator(LeafNode *leaf, slot_type slot, bool locked = false) : cur_(leaf), slot_(slot), locked_(locked) {}

    ~Iterator();

    explicit Iterator(const Iterator &);

    Iterator &operator=(const Iterator &lhs);

    Iterator(Iterator &&);

    Iterator &operator=(Iterator &&lhs);

    pointer operator->() const { return cur_->data_[cur_->slot_dir_[slot_]].get(); }

    reference operator*() const { return *(cur_->data_[cur_->slot_dir_[slot_]].get()); }

    const key_type &key() const { return cur_->keys_[cur_->slot_dir_[slot_]]; }

    value_type &value() const { return *(cur_->data_[cur_->slot_dir_[slot_]].get()); }

    // Prefix++
    Iterator &operator++();

    // Postfix++
    Iterator operator++(int);

    // Prefix--
    Iterator &operator--();

    // Postfix--
    Iterator operator--(int);

    bool operator==(const Iterator &x) const { return (x.cur_ == cur_) && (x.slot_ == slot_); }

    bool operator!=(const Iterator &x) const { return (x.cur_ != cur_) || (x.slot_ != slot_); }

   private:
    typename BPlusTree::LeafNode *cur_;
    slot_type slot_;
    bool locked_;
  };

  class ConstIterator : public std::iterator<std::bidirectional_iterator_tag, value_type> {
   public:
    using reference = BPlusTree::value_type &;
    using pointer = BPlusTree::value_type *;

    explicit ConstIterator(const BPlusTree *btree) : cur_(btree->leaf_nodes_.head), slot_(0), locked_(false) {}

    ~ConstIterator();

    ConstIterator(const LeafNode *leaf, slot_type slot, bool locked = false)
        : cur_(leaf), slot_(slot), locked_(locked) {}

    explicit ConstIterator(const ConstIterator &);

    ConstIterator &operator=(const ConstIterator &lhs);

    ConstIterator(ConstIterator &&);

    ConstIterator &operator=(ConstIterator &&lhs);

    pointer operator->() const { return cur_->data_[cur_->slot_dir_[slot_]].get(); }

    reference operator*() const { return *(cur_->data_[cur_->slot_dir_[slot_]].get()); }

    const key_type &key() const { return cur_->keys_[cur_->slot_dir_[slot_]]; }

    value_type &value() const { return *(cur_->data_[cur_->slot_dir_[slot_]].get()); }

    // Prefix++
    ConstIterator &operator++();

    // Postfix++
    ConstIterator operator++(int);

    // Prefix--
    ConstIterator &operator--();

    // Postfix--
    ConstIterator operator--(int);

    bool operator==(const ConstIterator &x) const { return (x.cur_ == cur_) && (x.slot_ == slot_); }

    bool operator!=(const ConstIterator &x) const { return (x.cur_ != cur_) || (x.slot_ != slot_); }

   private:
    const typename BPlusTree::LeafNode *cur_;
    slot_type slot_;
    bool locked_;
  };

  Iterator begin();

  Iterator end();

  ConstIterator begin() const;

  ConstIterator end() const;

  ConstIterator cbegin() const;

  ConstIterator cend() const;

  // Locate the entry with key
  ConstIterator Search(const key_type &key) const;
  Iterator Search(const key_type &key);

  value_type operator[](key_type key);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_INDEX_H_

#include "btree_impl.tpp"
#include "btree_iterator.tpp"
