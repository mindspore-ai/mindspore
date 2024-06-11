/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_PTR_LIST_REF_H
#define MINDSPORE_PI_JIT_PTR_LIST_REF_H
#include <iterator>
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {
template <typename T>
class PtrListNodeBase {
 public:
  PtrListNodeBase() = default;
  ~PtrListNodeBase() = default;
  T *GetPrev() const { return prev; }

  T *GetNext() const { return next; }

  void SetPrev(T *ptr) { prev = ptr; }

  void SetNext(T *ptr) { next = ptr; }

 private:
  T *prev = nullptr;
  T *next = nullptr;
};

// wrap iterator to run it backwards
template <typename T>
class ReversePtrListRefIterator {
 public:
  using iterator_category = typename std::iterator_traits<T>::iterator_category;
  using value_type = typename std::iterator_traits<T>::value_type;
  using difference_type = typename std::iterator_traits<T>::difference_type;
  using pointer = typename std::iterator_traits<T>::pointer;
  using reference = typename std::iterator_traits<T>::reference;

  using iterator_type = T;

  ReversePtrListRefIterator() : current() {}

  explicit ReversePtrListRefIterator(T right) : current(right) {}

  template <class Other>
  ReversePtrListRefIterator(const ReversePtrListRefIterator<Other> &right) : current(right.base()) {}

  template <class Other>
  ReversePtrListRefIterator &operator=(const ReversePtrListRefIterator<Other> &right) {
    current = right.base();
    return (*this);
  }

  ~ReversePtrListRefIterator() = default;

  T base() const { return current; }

  reference operator*() const { return *current; }

  pointer operator->() const { return &(operator*()); }

  ReversePtrListRefIterator &operator++() {
    --current;
    return (*this);
  }

  ReversePtrListRefIterator operator++(int) {
    ReversePtrListRefIterator tmp = *this;
    --current;
    return (tmp);
  }

  ReversePtrListRefIterator &operator--() {
    ++current;
    return (*this);
  }

  ReversePtrListRefIterator operator--(int) {
    ReversePtrListRefIterator tmp = *this;
    ++current;
    return (tmp);
  }

  bool operator==(const ReversePtrListRefIterator &Iterator) const { return this->base() == Iterator.base(); }

  bool operator!=(const ReversePtrListRefIterator &Iterator) const { return !(*this == Iterator); }

 protected:
  T current;
};

template <typename T>
class PtrListRefIterator {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;
  using const_pointer = const T *;
  using const_reference = const T &;

  PtrListRefIterator() = default;

  explicit PtrListRefIterator(pointer list_ref_iterator_ptr) : ptr(list_ref_iterator_ptr) {}

  template <typename U, typename = std::enable_if_t<std::is_same<U, std::remove_const_t<T>>::value>>
  PtrListRefIterator(const PtrListRefIterator<U> &ListIter_) : ptr(ListIter_.d()) {}

  ~PtrListRefIterator() = default;

  pointer d() const { return ptr; }

  reference operator*() const { return *ptr; }

  pointer operator->() const { return ptr; }

  PtrListRefIterator &operator++() {
    this->ptr = this->ptr->GetNext();
    return *this;
  }

  PtrListRefIterator &operator--() {
    this->ptr = this->ptr->GetPrev();
    return *this;
  }

  PtrListRefIterator operator++(int) {
    PtrListRefIterator it = *this;
    ++(*this);
    return it;
  }

  PtrListRefIterator operator--(int) {
    PtrListRefIterator it = *this;
    --(*this);
    return it;
  }

  bool operator==(const PtrListRefIterator &Iterator) const { return this->ptr == Iterator.ptr; }

  bool operator!=(const PtrListRefIterator &Iterator) const { return !(*this == Iterator); }

 private:
  pointer ptr = nullptr;
};

template <typename T>
class PtrListRef {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  using iterator = PtrListRefIterator<T>;
  using const_iterator = PtrListRefIterator<const T>;
  using reverse_iterator = ReversePtrListRefIterator<iterator>;
  using const_reverse_iterator = ReversePtrListRefIterator<const_iterator>;

  PtrListRef() = default;
  explicit PtrListRef(pointer list_value) : first(list_value), last(list_value) {}

  PtrListRef(pointer FirstList_, pointer LastList_)
      : first(FirstList_), last(LastList_ == nullptr ? FirstList_ : LastList_) {}

  ~PtrListRef() = default;

  iterator begin() { return iterator(this->first); }

  const_iterator begin() const { return const_iterator(this->first); }

  const_iterator cbegin() const { return const_iterator(this->first); }

  iterator end() { return iterator(this->last == nullptr ? nullptr : this->last->GetNext()); }

  const_iterator end() const { return const_iterator(this->last == nullptr ? nullptr : this->last->GetNext()); }

  const_iterator cend() const { return const_iterator(this->last == nullptr ? nullptr : this->last->GetNext()); }

  reverse_iterator rbegin() { return reverse_iterator(iterator(this->last)); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(const_iterator(this->last)); }

  const_reverse_iterator crbegin() const { return const_reverse_iterator(const_iterator(this->last)); }

  reverse_iterator rend() {
    return reverse_iterator(iterator(this->first == nullptr ? nullptr : this->first->GetPrev()));
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(const_iterator(this->first == nullptr ? nullptr : this->first->GetPrev()));
  }

  const_reverse_iterator crend() const {
    return const_reverse_iterator(const_iterator(this->first == nullptr ? nullptr : this->first->GetPrev()));
  }

  reference front() { return *(this->first); }

  reference back() { return *(this->last); }

  const_reference front() const { return *(this->first); }

  const_reference back() const { return *(this->last); }

  bool empty() const { return first == nullptr; }

  void update_front(pointer list_value) {
    if (list_value != nullptr) {
      list_value->SetPrev(nullptr);
    }
    this->first = list_value;
  }

  void push_front(pointer list_value) {
    if (this->last == nullptr) {
      this->first = list_value;
      this->last = list_value;
      list_value->SetPrev(nullptr);
      list_value->SetNext(nullptr);
    } else {
      MS_ASSERT(this->first != nullptr);
      this->first->SetPrev(list_value);
      list_value->SetPrev(nullptr);
      list_value->SetNext(this->first);
      this->first = list_value;
    }
  }

  void pop_front() {
    if (this->first == nullptr) {
      return;
    }

    this->first = this->first->GetNext();
    if (this->first != nullptr) {
      this->first->SetPrev(nullptr);
    }
  }

  void update_back(pointer list_value) {
    if (list_value != nullptr) {
      list_value->SetNext(nullptr);
    }
    this->last = list_value;
  }

  void push_back(pointer list_value) {
    if (this->last == nullptr) {
      this->first = list_value;
      this->last = list_value;
      list_value->SetPrev(nullptr);
    } else {
      this->last->SetNext(list_value);
      list_value->SetPrev(this->last);
      this->last = list_value;
    }
    list_value->SetNext(nullptr);
  }

  void pop_back() {
    if (this->last == nullptr) {
      return;
    }

    if (this->last->GetPrev() == nullptr) {
      this->first = nullptr;
      this->last = nullptr;
    } else {
      this->last = this->last->GetPrev();
      this->last->SetNext(nullptr);
    }
  }

  void insert(const_iterator list_where, pointer list_value) {
    if (list_where == const_iterator(this->first)) {
      this->push_front(list_value);
    } else if (list_where == this->cend()) {
      this->push_back(list_value);
    } else {
      // `list_where` stands for the position, however we made the data and node combined, so a const_cast is needed.
      auto *ptr = const_cast<T *>(&*list_where);
      list_value->SetPrev(ptr->GetPrev());
      list_value->SetNext(ptr);
      list_value->GetPrev()->SetNext(list_value);
      ptr->SetPrev(list_value);
    }
  }

  void insert(const_pointer list_where, pointer list_value) { this->insert(const_iterator(list_where), list_value); }

  // cut list two half, list_where is head of second half
  PtrListRef CutList(pointer list_where) {
    MS_ASSERT(!list_where || list_where == this->first || this->first == this->last);
    PtrListRef other = {const_cast<T *>(list_where), this->last};
    this->last = list_where->GetPrev();
    other.front().SetPrev(nullptr);
    this->last->SetNext(nullptr);
    return other;
  }

  PtrListRef CutList(iterator list_where) { return CutList(*list_where); }

  void insertAfter(const_iterator list_where, pointer list_value) {
    if (list_where == const_iterator(nullptr)) {
      this->push_front(list_value);
    } else if (list_where == const_iterator(this->last)) {
      this->push_back(list_value);
    } else {
      // `list_where` stands for the position, however we made the data and node combined, so a const_cast is needed.
      auto *ptr = const_cast<T *>(&*list_where);
      list_value->SetPrev(ptr);
      list_value->SetNext(ptr->GetNext());
      list_value->GetNext()->SetPrev(list_value);
      ptr->SetNext(list_value);
    }
  }

  void insertAfter(const_pointer list_where, pointer list_value) {
    this->insertAfter(const_iterator(list_where), list_value);
  }

  // clear other
  void splice(const_iterator list_where, PtrListRef *other) {
    if (other->empty()) {
      return;
    }
    MS_ASSERT(other->first && !other->first->GetPrev() && other->last && !other->last->GetNext());
    if (empty()) {
      this->first = other->first;
      this->last = other->last;
      other->clear();
      return;
    }
    if (list_where == this->end()) {
      this->last->SetNext(other->first);
      other->first->SetPrev(this->first);
      this->last = other->last;
      other->clear();
      return;
    }
    auto *ptr = const_cast<T *>(&*list_where);
    if (list_where == this->begin()) {
      this->first = other->first;
    } else {
      list_where->GetPrev()->SetNext(other->first);
      other->first->SetPrev(list_where->GetPrev());
    }
    ptr->SetPrev(other->last);
    other->last->SetNext(ptr);
    other->clear();
  }

  void splice(const_pointer list_where, PtrListRef *other) { listSplice(const_iterator(list_where), other); }

  void clear() {
    this->first = nullptr;
    this->last = nullptr;
  }

  iterator erase(const_iterator list_where) {
    if (list_where == this->cbegin() && list_where == this->rbegin().base()) {
      this->first = nullptr;
      this->last = nullptr;
    } else if (list_where == this->cbegin()) {
      // `list_where` stands for the position, however we made the data and node combined, so a const_cast is needed.
      auto *ptr = const_cast<T *>(&*list_where);
      this->first = ptr->GetNext();
      MS_ASSERT(this->first != nullptr);
      this->first->SetPrev(nullptr);
    } else if (list_where == this->rbegin().base()) {
      pop_back();
    } else {
      MS_ASSERT(list_where->GetPrev() != nullptr);
      // `list_where` stands for the position, however we made the data and node combined, so a const_cast is needed.
      auto *ptr = const_cast<T *>(&*list_where);
      ptr->GetPrev()->SetNext(ptr->GetNext());
      if (ptr->GetNext()) {
        ptr->GetNext()->SetPrev(ptr->GetPrev());
      }
    }
    return iterator(nullptr);
  }

  iterator erase(const_pointer list_where) { return this->erase(const_iterator(list_where)); }

  void set_first(T *f) { this->first = f; }

  void set_last(T *f) { this->last = f; }

 private:
  T *first = nullptr;
  T *last = nullptr;
};

template <typename Iterator>
auto to_ptr(Iterator it) -> typename std::iterator_traits<Iterator>::pointer {
  return it.d();
}

template <typename Iterator>
auto to_ptr(ReversePtrListRefIterator<Iterator> it) -> typename std::iterator_traits<Iterator>::pointer {
  return it.base().d();
}
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_PTR_LIST_REF_H
