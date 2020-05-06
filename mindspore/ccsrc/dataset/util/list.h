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
#ifndef DATASET_UTIL_LIST_H_
#define DATASET_UTIL_LIST_H_

#include <iostream>
#include <iterator>
#include "dataset/util/de_error.h"

namespace mindspore {
namespace dataset {
template <typename T>
struct Node {
  using value_type = T;
  using pointer = T *;
  pointer prev;
  pointer next;

  Node() {
    prev = nullptr;
    next = nullptr;
  }
};

template <typename T>
struct List {
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  int count;
  pointer head;
  pointer tail;
  Node<T> T::*node;

  // Constructor
  explicit List(Node<T> T::*m) : count(0), head(nullptr), tail(nullptr), node(m) {}

  // Destructor
  virtual ~List() {
    head = nullptr;
    tail = nullptr;
  }

  // Prepend
  virtual void Prepend(pointer elem) {
    Node<T> &elem_node = elem->*node;
    elem_node.prev = nullptr;
    elem_node.next = head;
    if (head != nullptr) {
      Node<T> &base_node = head->*node;
      base_node.prev = elem;
    }
    head = elem;
    if (tail == nullptr) {
      tail = elem;
    }
    ++count;
  }

  // Append
  virtual void Append(pointer elem) {
    Node<T> &elem_node = elem->*node;
    elem_node.next = nullptr;
    elem_node.prev = tail;
    if (tail != nullptr) {
      Node<T> &base_node = tail->*node;
      base_node.next = elem;
    }
    tail = elem;
    if (head == nullptr) {
      head = elem;
    }
    ++count;
  }

  // Insert elem2 after elem1 in the list.
  virtual void InsertAfter(pointer elem1, pointer elem2) {
    DS_ASSERT(elem1 != elem2);
    Node<T> &elem1_node = elem1->*node;
    Node<T> &elem2_node = elem2->*node;
    elem2_node.prev = elem1;
    elem2_node.next = elem1_node.next;
    if (elem1_node.next != nullptr) {
      Node<T> &next_node = elem1_node.next->*node;
      next_node.prev = elem2;
    }
    elem1_node.next = elem2;
    if (tail == elem1) {
      tail = elem2;
    }
    ++count;
  }

  // Remove an element in the list
  virtual void Remove(pointer elem) noexcept {
    Node<T> &elem_node = elem->*node;
    if (elem_node.next != nullptr) {
      Node<T> &next_node = elem_node.next->*node;
      next_node.prev = elem_node.prev;
    } else {
      tail = elem_node.prev;
    }
    if (elem_node.prev != nullptr) {
      Node<T> &prev_node = elem_node.prev->*node;
      prev_node.next = elem_node.next;
    } else {
      head = elem_node.next;
    }
    elem_node.prev = nullptr;
    elem_node.next = nullptr;
    --count;
  }

  // Iterator
  class Iterator : public std::iterator<std::forward_iterator_tag, T> {
   public:
    pointer elem_;

    explicit Iterator(const List<T> &v, pointer p = nullptr) : elem_(p), li_(v) {}

    ~Iterator() = default;

    reference operator*() { return *elem_; }

    pointer operator->() { return elem_; }

    const_reference operator*() const { return *elem_; }

    const_pointer operator->() const { return elem_; }

    bool operator==(const Iterator &rhs) const { return elem_ == rhs.elem_; }

    bool operator!=(const Iterator &rhs) const { return elem_ != rhs.elem_; }

    // Prefix increment
    Iterator &operator++() {
      Node<T> &elem_node = elem_->*(li_.node);
      elem_ = elem_node.next;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int junk) {
      Iterator tmp(*this);
      Node<T> &elem_node = elem_->*(li_.node);
      elem_ = elem_node.next;
      return tmp;
    }

    // Prefix decrement
    Iterator &operator--() {
      Node<T> &elem_node = elem_->*(li_.node);
      elem_ = elem_node.prev;
      return *this;
    }

    // Postfix decrement
    Iterator operator--(int junk) {
      Iterator tmp(*this);
      Node<T> &elem_node = elem_->*(li_.node);
      elem_ = elem_node.prev;
      return tmp;
    }

   private:
    const List<T> &li_;
  };

  Iterator begin() {
    Iterator it(*this, head);
    return it;
  }

  Iterator end() {
    Iterator it(*this);
    return it;
  }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_LIST_H_
