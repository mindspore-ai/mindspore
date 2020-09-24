/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_INTERNAL_INCLUDE_VECTOR_H
#define MINDSPORE_LITE_INTERNAL_INCLUDE_VECTOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#define DEFAULT_CAPACITY 4

struct MSTensor;
struct Node;

template <typename T>
class Vector {
 public:
  Vector();

  explicit Vector(size_t size);

  Vector(size_t size, const T &value);

  Vector(const Vector<T> &vector);

  ~Vector();

  void clear();

  void push_back(const T &elem);

  void push_back(T &&);

  void pop_back();

  void insert(const T &elem, size_t index);

  T *begin();

  const T *begin() const;

  T *end();

  const T *end() const;

  T &front();

  const T &front() const;

  T &back();

  const T &back() const;

  T &at(size_t index);

  const T &at(size_t index) const;

  T &operator[](size_t index);

  const T &operator[](size_t index) const;

  T *data();

  const T *data() const;

  size_t size() const;

  size_t capacity() const;

  bool empty() const;

  void erase(size_t index);

  void resize(size_t size);

  void reserve(size_t capacity);

  Vector<T> &operator=(const Vector<T> &v);

 private:
  size_t size_;
  size_t elem_size_;
  size_t capacity_;
  T *data_;
};

template <typename T>
bool operator==(const Vector<T> &lhs, const Vector<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool operator!=(const Vector<T> &lhs, const Vector<T> &rhs) {
  return !(lhs == rhs);
}
#endif  // MINDSPORE_LITE_INTERNAL_INCLUDE_VECTOR_H
