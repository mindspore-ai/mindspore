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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TREAP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TREAP_H_

#include <functional>
#include <iterator>
#include <stack>
#include <utility>
#include <vector>

namespace mindspore {
namespace dataset {
// A treap is a combination of binary search tree and heap. Each key is given a priority. The priority
// for any non-leaf node is greater than or equal to the priority of its children.
// @tparam K
//  Data type of key
// @tparam P
//  Data type of priority
// @tparam KC
//  Class to compare key. Default to std::less
// @tparam KP
//  Class to compare priority. Default to std:less
template <typename K, typename P, typename KC = std::less<K>, typename KP = std::less<P>>
class Treap {
 public:
  using key_type = K;
  using priority_type = P;
  using key_compare = KC;
  using priority_compare = KP;

  struct NodeValue {
    key_type key;
    priority_type priority;
  };

  class TreapNode {
   public:
    TreapNode() : left(nullptr), right(nullptr) {}
    ~TreapNode() {
      left = nullptr;
      right = nullptr;
    }
    NodeValue nv;
    TreapNode *left;
    TreapNode *right;
  };

  // search API
  // @param k
  //    key to search for
  // @return
  //    a pair is returned. The 2nd value of type bool indicate if the search is successful.
  //    If true, the first value of the pair contains the key and the priority.
  std::pair<NodeValue, bool> Search(key_type k) const {
    auto *n = Search(root_, k);
    if (n != nullptr) {
      return std::make_pair(n->nv, true);
    } else {
      return std::make_pair(NodeValue{key_type(), priority_type()}, false);
    }
  }

  // @return
  //    Return the root of the heap. It has the highest priority. But not necessarily the first key.
  std::pair<NodeValue, bool> Top() const {
    if (root_ != nullptr) {
      return std::make_pair(root_->nv, true);
    } else {
      return std::make_pair(NodeValue{key_type(), priority_type()}, false);
    }
  }

  // Remove the root of the heap.
  void Pop() {
    if (root_ != nullptr) {
      DeleteKey(root_->nv.key);
    }
  }

  // Insert API.
  // @param k
  //    The key to insert.
  // @param p
  //    The priority of the key.
  void Insert(key_type k, priority_type p) { root_ = Insert(root_, k, p); }

  // Delete a key.
  // @param k
  void DeleteKey(key_type k) { root_ = DeleteNode(root_, k); }

  Treap() : root_(nullptr), count_(0) { free_list_.reserve(kResvSz); }

  ~Treap() noexcept {
    DeleteTreap(root_);
    while (!free_list_.empty()) {
      TreapNode *n = free_list_.back();
      delete (n);
      free_list_.pop_back();
    }
  }

  class iterator : public std::iterator<std::forward_iterator_tag, TreapNode> {
   public:
    explicit iterator(Treap *tr) : tr_(tr), cur_(nullptr) {
      if (tr_ != nullptr) {
        cur_ = tr_->root_;
        while (cur_ != nullptr) {
          stack_.push(cur_);
          cur_ = cur_->left;
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
    }
    ~iterator() {
      tr_ = nullptr;
      cur_ = nullptr;
    }

    NodeValue &operator*() { return cur_->nv; }

    NodeValue *operator->() { return &(cur_->nv); }

    const TreapNode &operator*() const { return *cur_; }

    const TreapNode *operator->() const { return cur_; }

    bool operator==(const iterator &rhs) const { return cur_ == rhs.cur_; }

    bool operator!=(const iterator &rhs) const { return cur_ != rhs.cur_; }

    // Prefix increment
    iterator &operator++() {
      if (cur_) {
        stack_.pop();
        if (cur_->right) {
          TreapNode *n = cur_->right;
          while (n) {
            stack_.push(n);
            n = n->left;
          }
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
      return *this;
    }

    // Postfix increment
    iterator operator++(int junk) {
      iterator tmp(*this);
      if (cur_) {
        stack_.pop();
        if (cur_->right) {
          TreapNode *n = cur_->right;
          while (n) {
            stack_.push(n);
            n = n->left;
          }
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
      return tmp;
    }

   private:
    Treap *tr_;
    TreapNode *cur_;
    std::stack<TreapNode *> stack_;
  };

  class const_iterator : public std::iterator<std::forward_iterator_tag, TreapNode> {
   public:
    explicit const_iterator(const Treap *tr) : tr_(tr), cur_(nullptr) {
      if (tr_ != nullptr) {
        cur_ = tr_->root_;
        while (cur_ != nullptr) {
          stack_.push(cur_);
          cur_ = cur_->left;
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
    }
    ~const_iterator() {
      tr_ = nullptr;
      cur_ = nullptr;
    }

    const NodeValue &operator*() const { return cur_->nv; }

    const NodeValue *operator->() const { return &(cur_->nv); }

    bool operator==(const const_iterator &rhs) const { return cur_ == rhs.cur_; }

    bool operator!=(const const_iterator &rhs) const { return cur_ != rhs.cur_; }

    // Prefix increment
    const_iterator &operator++() {
      if (cur_) {
        stack_.pop();
        if (cur_->right != nullptr) {
          TreapNode *n = cur_->right;
          while (n) {
            stack_.push(n);
            n = n->left;
          }
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
      return *this;
    }

    // Postfix increment
    const_iterator operator++(int junk) {
      iterator tmp(*this);
      if (cur_) {
        stack_.pop();
        if ((cur_->right) != nullptr) {
          TreapNode *n = cur_->right;
          while (n) {
            stack_.push(n);
            n = n->left;
          }
        }
      }
      if (!stack_.empty()) {
        cur_ = stack_.top();
      } else {
        cur_ = nullptr;
      }
      return tmp;
    }

   private:
    const Treap *tr_;
    TreapNode *cur_;
    std::stack<TreapNode *> stack_;
  };

  iterator begin() { return iterator(this); }

  iterator end() { return iterator(nullptr); }

  const_iterator begin() const { return const_iterator(this); }

  const_iterator end() const { return const_iterator(nullptr); }

  const_iterator cbegin() { return const_iterator(this); }

  const_iterator cend() { return const_iterator(nullptr); }

  bool empty() { return root_ == nullptr; }

  size_t size() { return count_; }

 private:
  TreapNode *NewNode() {
    TreapNode *n = nullptr;
    if (!free_list_.empty()) {
      n = free_list_.back();
      free_list_.pop_back();
      new (n) TreapNode();
    } else {
      n = new TreapNode();
    }
    return n;
  }

  void FreeNode(TreapNode *n) { free_list_.push_back(n); }

  void DeleteTreap(TreapNode *n) noexcept {
    if (n == nullptr) {
      return;
    }
    TreapNode *x = n->left;
    TreapNode *y = n->right;
    delete (n);
    DeleteTreap(x);
    DeleteTreap(y);
  }

  TreapNode *RightRotate(TreapNode *y) {
    TreapNode *x = y->left;
    TreapNode *T2 = x->right;
    x->right = y;
    y->left = T2;
    return x;
  }

  TreapNode *LeftRotate(TreapNode *x) {
    TreapNode *y = x->right;
    TreapNode *T2 = y->left;
    y->left = x;
    x->right = T2;
    return y;
  }

  TreapNode *Search(TreapNode *n, key_type k) const {
    key_compare keyCompare;
    if (n == nullptr) {
      return n;
    } else if (keyCompare(k, n->nv.key)) {
      return Search(n->left, k);
    } else if (keyCompare(n->nv.key, k)) {
      return Search(n->right, k);
    } else {
      return n;
    }
  }

  TreapNode *Insert(TreapNode *n, key_type k, priority_type p) {
    key_compare keyCompare;
    priority_compare priorityCompare;
    if (n == nullptr) {
      n = NewNode();
      n->nv.key = k;
      n->nv.priority = p;
      count_++;
      return n;
    }
    if (keyCompare(k, n->nv.key)) {
      n->left = Insert(n->left, k, p);
      if (priorityCompare(n->nv.priority, n->left->nv.priority)) {
        n = RightRotate(n);
      }
    } else if (keyCompare(n->nv.key, k)) {
      n->right = Insert(n->right, k, p);
      if (priorityCompare(n->nv.priority, n->right->nv.priority)) {
        n = LeftRotate(n);
      }
    } else {
      // If we insert the same key again, do nothing.
      return n;
    }
    return n;
  }

  TreapNode *DeleteNode(TreapNode *n, key_type k) {
    key_compare keyCompare;
    priority_compare priorityCompare;
    if (n == nullptr) {
      return n;
    }
    if (keyCompare(k, n->nv.key)) {
      n->left = DeleteNode(n->left, k);
    } else if (keyCompare(n->nv.key, k)) {
      n->right = DeleteNode(n->right, k);
    } else if (n->left == nullptr) {
      TreapNode *t = n;
      n = n->right;
      FreeNode(t);
      count_--;
    } else if (n->right == nullptr) {
      TreapNode *t = n;
      n = n->left;
      FreeNode(t);
      count_--;
    } else if (priorityCompare(n->left->nv.priority, n->right->nv.priority)) {
      n = LeftRotate(n);
      n->left = DeleteNode(n->left, k);
    } else {
      n = RightRotate(n);
      n->right = DeleteNode(n->right, k);
    }
    return n;
  }

  static constexpr int kResvSz = 512;
  TreapNode *root_;
  size_t count_;
  std::vector<TreapNode *> free_list_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TREAP_H_
