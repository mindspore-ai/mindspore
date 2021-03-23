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

#ifndef MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
#define MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_

#ifndef NOT_USE_STL
#include <vector>
#include <string>
#include <memory>
#include <functional>
#else
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <new>
#endif  // NOT_USE_STL

#ifndef MS_API
#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif
#endif

namespace mindspore {
namespace schema {
struct Tensor;
}  // namespace schema

namespace tensor {
class MSTensor;
}  // namespace tensor

namespace lite {
struct DeviceContext;
}  // namespace lite

#ifdef NOT_USE_STL
class String {
 public:
  String();
  String(size_t count, char ch);
  String(const char *s, size_t count);
  explicit String(const char *s);
  String(const String &other);
  String(const String &other, size_t pos, size_t count = npos);

  ~String();

  String &operator=(const String &str);
  String &operator=(const char *str);

  char &at(size_t pos);
  const char &at(size_t pos) const;
  inline char &operator[](size_t pos);
  inline const char &operator[](size_t pos) const;
  char *data() noexcept;
  const char *data() const noexcept;
  const char *c_str() const noexcept;

  // capacity
  bool empty() const noexcept;
  size_t size() const noexcept;
  size_t length() const noexcept;

  // operations
  void clear() noexcept;
  String &append(size_t count, const char ch);
  String &append(const String &str);
  String &append(const char *s);
  String &operator+(const String &str);
  String &operator+=(const String &str);
  String &operator+=(const char *str);
  String &operator+=(const char ch);
  int compare(const String &str) const;
  int compare(const char *str) const;

  String substr(size_t pos = 0, size_t count = npos) const;

  static const size_t npos = -1;

 private:
  size_t size_;
  char *buffer_;
};

String operator+(const String &lhs, const char *rhs);
String operator+(const char *lhs, const String &rhs);

bool operator==(const String &lhs, const String &rhs);
bool operator==(const String &lhs, const char *rhs);
bool operator==(const char *lhs, const String &rhs);

String to_string(int32_t value);
String to_string(float value);

#define DEFAULT_CAPACITY 4
#define MS_C_EXCEPTION(...) exit(1)
#define MIN(x, y) ((x < y) ? (x) : (y))
template <typename T>
class Vector {
 public:
  Vector() {
    size_ = 0;
    capacity_ = DEFAULT_CAPACITY;
    elem_size_ = sizeof(T);
    data_ = nullptr;
  }

  explicit Vector(size_t size) {
    size_ = size;
    elem_size_ = sizeof(T);
    capacity_ = (size == 0 ? DEFAULT_CAPACITY : size);
    data_ = new (std::nothrow) T[capacity_];
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
  }

  Vector(size_t size, const T &value) {
    size_ = size;
    elem_size_ = sizeof(T);
    capacity_ = (size == 0 ? DEFAULT_CAPACITY : size);
    data_ = new (std::nothrow) T[capacity_];
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      data_[i] = value;
    }
  }

  Vector(const Vector<T> &vec) {
    size_ = vec.size_;
    elem_size_ = sizeof(T);
    capacity_ = vec.capacity_;
    data_ = new (std::nothrow) T[capacity_];
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      data_[i] = vec.data_[i];
    }
  }

  ~Vector() {
    if (data_ != nullptr) {
      delete[] data_;
    }
  }

  void clear() {
    size_ = 0;
    if (data_ != nullptr) {
      delete[] data_;
      data_ = nullptr;
    }
  }

  void push_back(const T &elem) {
    if (data_ == nullptr) {
      data_ = new (std::nothrow) T[capacity_];
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

  void push_back(T &&elem) {
    if (data_ == nullptr) {
      data_ = new (std::nothrow) T[capacity_];
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

  void pop_back() {
    if (size_ > 0) {
      --size_;
    } else {
      MS_C_EXCEPTION("Index is out of range!");
    }
  }

  void insert(const T &elem, size_t index) {
    if (index <= size_) {
      ++size_;
      if (size_ > capacity_) {
        resize(size_);
      }
      if (index == size_ - 1) {
        push_back(elem);
      } else {
        for (int i = static_cast<int>(size_) - 1; i > static_cast<int>(index); --i) {
          data_[i + 1] = data_[i];
        }
        data_[index] = elem;
      }
    } else {
      MS_C_EXCEPTION("Input index is out of range!");
    }
  }

  T *begin() { return data_; }

  const T *begin() const { return data_; }

  T *end() { return data_ + size_; }

  const T *end() const { return data_ + size_; }

  T &front() {
    if (size_ > 0) {
      return data_[0];
    }
    MS_C_EXCEPTION("Index is out of range!");
  }

  const T &front() const {
    if (size_ > 0) {
      return data_[0];
    }
    MS_C_EXCEPTION("Index is out of range!");
  }

  T &back() {
    if (size_ > 0) {
      return data_[size_ - 1];
    }
    MS_C_EXCEPTION("Index is out of range!");
  }

  const T &back() const {
    if (size_ > 0) {
      return data_[size_ - 1];
    }
    MS_C_EXCEPTION("Index is out of range!");
  }

  T &at(size_t index) {
    if (index < size_) {
      return data_[index];
    }
    MS_C_EXCEPTION("Input index is out of range!");
  }

  const T &at(size_t index) const {
    if (index < size_) {
      return data_[index];
    }
    MS_C_EXCEPTION("Input index is out of range!");
  }

  T &operator[](size_t index) {
    if (index < size_) {
      return data_[index];
    }
    MS_C_EXCEPTION("Input index is out of range!");
  }

  const T &operator[](size_t index) const {
    if (index < size_) {
      return data_[index];
    }
    MS_C_EXCEPTION("Input index is out of range!");
  }

  T *data() { return data_; }

  const T *data() const { return data_; }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  bool empty() const { return size_ == 0; }

  void erase(size_t index) {
    if (index == size_ - 1) {
      --size_;
    } else if (index < size_) {
      for (int i = index; i < static_cast<int>(size_); ++i) {
        data_[i] = data_[i + 1];
      }
      --size_;
    } else {
      MS_C_EXCEPTION("Input index is out of range!");
    }
  }

  void resize(size_t size) {
    while (size > capacity_) {
      capacity_ *= 2;
    }
    T *tmp = data_;
    data_ = new (std::nothrow) T[capacity_];
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    for (int i = 0; i < MIN(static_cast<int>(size), static_cast<int>(size_)); ++i) {
      data_[i] = tmp[i];
    }
    size_ = size;
    delete[] tmp;
  }

  void reserve(size_t capacity) {
    if (capacity > capacity_) {
      capacity_ = capacity;
    }
  }

  Vector<T> &operator=(const Vector<T> &vec) {
    if (this == &vec) {
      return *this;
    }
    size_ = vec.size_;
    elem_size_ = sizeof(T);
    capacity_ = vec.capacity_;
    data_ = new (std::nothrow) T[capacity_];
    if (data_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      data_[i] = vec.data_[i];
    }
    return *this;
  }

 private:
  size_t size_;
  size_t elem_size_;
  size_t capacity_;
  T *data_;
};
using TensorPtrVector = Vector<mindspore::schema::Tensor *>;
using Uint32Vector = Vector<uint32_t>;
using AllocatorPtr = void *;
using DeviceContextVector = Vector<lite::DeviceContext>;
using KernelCallBack = void (*)(void *, void *);
#else
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
using AllocatorPtr = std::shared_ptr<Allocator>;

using TensorPtrVector = std::vector<mindspore::schema::Tensor *>;
using Uint32Vector = std::vector<uint32_t>;
template <typename T>
using Vector = std::vector<T>;

template <typename T>
inline std::string to_string(T t) {
  return std::to_string(t);
}

namespace tensor {
using String = std::string;
}  // namespace tensor

namespace session {
using String = std::string;
}  // namespace session

/// \brief CallBackParam defined input arguments for callBack function.
struct CallBackParam {
  session::String node_name; /**< node name argument */
  session::String node_type; /**< node type argument */
};

struct GPUCallBackParam : CallBackParam {
  double execute_time{-1.f};
};

/// \brief KernelCallBack defined the function pointer for callBack.
using KernelCallBack = std::function<bool(Vector<tensor::MSTensor *> inputs, Vector<tensor::MSTensor *> outputs,
                                          const CallBackParam &opInfo)>;

namespace lite {
using String = std::string;
using DeviceContextVector = std::vector<DeviceContext>;

/// \brief Set data of MSTensor from string vector.
///
/// \param[in] input string vector.
/// \param[out] MSTensor.
///
/// \return STATUS as an error code of this interface, STATUS is defined in errorcode.h.
int MS_API StringsToMSTensor(const Vector<String> &inputs, tensor::MSTensor *tensor);

/// \brief Get string vector from MSTensor.
/// \param[in] MSTensor.
/// \return string vector.
Vector<String> MS_API MSTensorToStrings(const tensor::MSTensor *tensor);
}  // namespace lite
#endif  // NOT_USE_STL
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
