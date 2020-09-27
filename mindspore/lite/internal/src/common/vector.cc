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
#include "internal/include/vector.h"
#include "internal/include/string.h"
#include "internal/src/lite_log.h"

#define MIN(x, y) ((x < y) ? (x) : (y))

template <typename T>
Vector<T>::Vector() {
  size_ = 0;
  capacity_ = DEFAULT_CAPACITY;
  elem_size_ = sizeof(T);
  data_ = nullptr;
}

template <typename T>
Vector<T>::Vector(size_t size) {
  size_ = size;
  elem_size_ = sizeof(T);
  capacity_ = (size == 0 ? DEFAULT_CAPACITY : size);
  data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
  if (data_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memset(data_, 0, capacity_ * elem_size_);
}

template <typename T>
Vector<T>::Vector(size_t size, const T &value) {
  size_ = size;
  elem_size_ = sizeof(T);
  capacity_ = (size == 0 ? DEFAULT_CAPACITY : size);
  data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
  if (data_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  for (int i = 0; i < size; ++i) {
    data_[i] = value;
  }
}

template <typename T>
Vector<T>::Vector(const Vector<T> &vec) {
  size_ = vec.size_;
  elem_size_ = sizeof(T);
  capacity_ = vec.capacity_;
  data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
  if (data_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(data_, vec.data_, size_ * elem_size_);
}

template <typename T>
Vector<T> &Vector<T>::operator=(const Vector<T> &vec) {
  if (this == &vec) {
    return *this;
  }
  size_ = vec.size_;
  elem_size_ = sizeof(T);
  capacity_ = vec.capacity_;
  data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
  if (data_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(data_, vec.data_, size_ * elem_size_);
  return *this;
}

template <typename T>
Vector<T>::~Vector() {
  if (data_ != nullptr) {
    free(data_);
  }
}

template <typename T>
void Vector<T>::clear() {
  size_ = 0;
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
}

template <typename T>
void Vector<T>::push_back(const T &elem) {
  if (data_ == nullptr) {
    data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
  } else if (size_ == capacity_) {
    resize(size_ + 1);
    --size_;
  }
  data_[size_] = elem;
  ++size_;
}

template <typename T>
void Vector<T>::push_back(T &&elem) {
  if (data_ == nullptr) {
    data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
  } else if (size_ == capacity_) {
    resize(size_ + 1);
    --size_;
  }
  data_[size_] = elem;
  ++size_;
}

template <typename T>
void Vector<T>::pop_back() {
  if (size_ > 0) {
    --size_;
  } else {
    MS_C_EXCEPTION("Index is out of range!");
  }
}

template <typename T>
void Vector<T>::insert(const T &elem, size_t index) {
  if (index <= size_) {
    ++size_;
    if (size_ > capacity_) {
      resize(size_);
    }
    if (index == size_ - 1) {
      push_back(elem);
    } else {
      memmove(data_ + index + 1, data_ + index, (size_ - index - 1) * elem_size_);
      data_[index] = elem;
    }
  } else {
    MS_C_EXCEPTION("Input index is out of range!");
  }
}

template <typename T>
T *Vector<T>::begin() {
  return data_;
}

template <typename T>
const T *Vector<T>::begin() const {
  return data_;
}

template <typename T>
T *Vector<T>::end() {
  return data_ + size_;
}

template <typename T>
const T *Vector<T>::end() const {
  return data_ + size_;
}

template <typename T>
T &Vector<T>::front() {
  if (size_ > 0) {
    return data_[0];
  }
  MS_C_EXCEPTION("Index is out of range!");
}

template <typename T>
const T &Vector<T>::front() const {
  if (size_ > 0) {
    return data_[0];
  }
  MS_C_EXCEPTION("Index is out of range!");
}
template <typename T>
T &Vector<T>::back() {
  if (size_ > 0) {
    return data_[size_ - 1];
  }
  MS_C_EXCEPTION("Index is out of range!");
}
template <typename T>
const T &Vector<T>::back() const {
  if (size_ > 0) {
    return data_[size_ - 1];
  }
  MS_C_EXCEPTION("Index is out of range!");
}

template <typename T>
T &Vector<T>::at(size_t index) {
  if (index < size_) {
    return data_[index];
  }
  MS_C_EXCEPTION("Input index is out of range!");
}

template <typename T>
const T &Vector<T>::at(size_t index) const {
  if (index < size_) {
    return data_[index];
  }
  MS_C_EXCEPTION("Input index is out of range!");
}

template <typename T>
T &Vector<T>::operator[](size_t index) {
  if (index < size_) {
    return data_[index];
  }
  MS_C_EXCEPTION("Input index is out of range!");
}

template <typename T>
const T &Vector<T>::operator[](size_t index) const {
  if (index < size_) {
    return data_[index];
  }
  MS_C_EXCEPTION("Input index is out of range!");
}

template <typename T>
T *Vector<T>::data() {
  return data_;
}

template <typename T>
const T *Vector<T>::data() const {
  return data_;
}

template <typename T>
size_t Vector<T>::size() const {
  return size_;
}

template <typename T>
size_t Vector<T>::capacity() const {
  return capacity_;
}

template <typename T>
bool Vector<T>::empty() const {
  return size_ == 0;
}

template <typename T>
void Vector<T>::erase(size_t index) {
  if (index == size_ - 1) {
    --size_;
  } else if (index < size_) {
    memmove(data_ + index, data_ + index + 1, (size_ - index - 1) * elem_size_);
    --size_;
  } else {
    MS_C_EXCEPTION("Input index is out of range!");
  }
}

template <typename T>
void Vector<T>::resize(size_t size) {
  while (size > capacity_) {
    capacity_ *= 2;
  }
  T *tmp = data_;
  data_ = reinterpret_cast<T *>(malloc(capacity_ * elem_size_));
  if (data_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(data_, tmp, MIN(size, size_) * elem_size_);
  size_ = size;
  free(tmp);
}

template <typename T>
void Vector<T>::reserve(size_t capacity) {
  if (capacity > capacity_) {
    capacity_ = capacity;
  }
}

template class Vector<int>;
template class Vector<Vector<int>>;
template class Vector<uint32_t>;
template class Vector<String>;
template class Vector<MSTensor *>;
template class Vector<Node *>;
