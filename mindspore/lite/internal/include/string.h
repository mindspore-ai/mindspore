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
#ifndef MINDSPORE_LITE_INTERNAL_SRC_STRING_H_
#define MINDSPORE_LITE_INTERNAL_SRC_STRING_H_
#include <string.h>
#include <stdint.h>

typedef struct String {
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
} String;

bool operator==(const String &lhs, const String &rhs);
bool operator==(const String &lhs, const char *rhs);
bool operator==(const char *lhs, const String rhs);

bool operator!=(const String &lhs, const String &rhs);
bool operator!=(const String &lhs, const char *rhs);
bool operator!=(const char *lhs, const String rhs);

bool operator<(const String &lhs, const String &rhs);
bool operator<(const String &lhs, const char *rhs);
bool operator<(const char *lhs, const String rhs);

bool operator>(const String &lhs, const String &rhs);
bool operator>(const String &lhs, const char *rhs);
bool operator>(const char *lhs, const String rhs);

bool operator<=(const String &lhs, const String &rhs);
bool operator<=(const String &lhs, const char *rhs);
bool operator<=(const char *lhs, const String rhs);

bool operator>=(const String &lhs, const String &rhs);
bool operator>=(const String &lhs, const char *rhs);
bool operator>=(const char *lhs, const String rhs);

String to_String(int32_t value);
String to_String(int64_t value);
String to_String(uint32_t value);
String to_String(uint64_t value);
String to_String(float value);
String to_String(double value);
String to_String(long double value);

#endif  // MINDSPORE_LITE_INTERNAL_SRC_STRING_H_
