/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/const_blocks/mstring.h"

namespace mindspore::lite::micro {

const char *string_source = R"RAW(

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifdef NOT_USE_STL
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include "include/lite_utils.h"

namespace mindspore {
String::String() {
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  buffer_[0] = '\0';
  size_ = 0;
}

String::String(size_t count, char ch) {
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (count + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memset(buffer_, ch, count);
  buffer_[count] = '\0';
  size_ = count;
}
String::String(const char *s, size_t count) {
  if (s == nullptr) {
    buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
    if (buffer_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    buffer_[0] = '\0';
    size_ = 0;
    return;
  }
  size_t size_s = strlen(s);
  if (size_s <= count) {
    size_ = size_s;
  } else {
    size_ = count;
  }
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (size_ + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  strncpy(buffer_, s, size_);
  buffer_[size_] = '\0';
}

String::String(const char *s) {
  if (s == nullptr) {
    buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
    if (buffer_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    buffer_[0] = '\0';
    size_ = 0;
    return;
  }
  size_ = strlen(s);
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (size_ + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(buffer_, s, size_ + 1);
}

String::String(const String &other) {
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (other.size_ + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  size_ = other.size_;
  memcpy(buffer_, other.buffer_, size_ + 1);
}

String::String(const String &other, size_t pos, size_t count) {
  if (pos >= other.size_) {
    buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
    if (buffer_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    buffer_[0] = '\0';
    size_ = 0;
  } else {
    if (count == npos) {
      count = other.size_ - pos;
    }
    if (pos + count > other.size_) {
      size_ = other.size_ - pos;
    } else {
      size_ = count;
    }
    buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (size_ + 1)));
    if (buffer_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    strncpy(buffer_, other.buffer_ + pos, size_);
    buffer_[size_] = '\0';
  }
}

String::~String() { free(buffer_); }

String &String::operator=(const String &str) {
  if (this == &str) {
    return *this;
  }
  free(buffer_);
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (str.size_ + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  size_ = str.size_;
  memcpy(buffer_, str.buffer_, size_ + 1);
  return *this;
}

String &String::operator=(const char *str) {
  free(buffer_);
  if (str == nullptr) {
    buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
    if (buffer_ == nullptr) {
      MS_C_EXCEPTION("malloc data failed");
    }
    buffer_[0] = '\0';
    size_ = 0;
    return *this;
  }
  size_t size_s = strlen(str);
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * (size_s + 1)));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  size_ = size_s;
  memcpy(buffer_, str, size_ + 1);
  return *this;
}

char &String::at(size_t pos) {
  if (pos >= size_) {
    MS_C_EXCEPTION("pos out of range");
  }
  return buffer_[pos];
}
const char &String::at(size_t pos) const {
  if (pos >= size_) {
    MS_C_EXCEPTION("pos out of range");
  }
  return buffer_[pos];
}
char &String::operator[](size_t pos) {
  if (pos >= size_) {
    MS_C_EXCEPTION("pos out of range");
  }
  return this->at(pos);
}
const char &String::operator[](size_t pos) const {
  if (pos >= size_) {
    MS_C_EXCEPTION("pos out of range");
  }
  return this->at(pos);
}
char *String::data() noexcept { return buffer_; };
const char *String::data() const noexcept { return buffer_; }
const char *String::c_str() const noexcept { return buffer_; }

// capacity
bool String::empty() const noexcept { return size_ == 0; }
size_t String::size() const noexcept { return size_; }
size_t String::length() const noexcept { return size_; }

// operations
void String::clear() noexcept {
  free(buffer_);
  buffer_ = reinterpret_cast<char *>(malloc(sizeof(char) * 1));
  if (buffer_ == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  buffer_[0] = '\0';
  size_ = 0;
}

String &String::operator+(const String &str) {
  (*this) += str;
  return *this;
}

String &String::operator+=(const String &str) {
  size_t new_size = size_ + str.size_;
  char *tmp = reinterpret_cast<char *>(malloc(sizeof(char) * (new_size + 1)));
  if (tmp == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(tmp, this->buffer_, size_ + 1);
  strncat(tmp, str.buffer_, str.size_);
  tmp[new_size] = '\0';
  free(buffer_);
  buffer_ = tmp;
  size_ = new_size;
  return *this;
}

String &String::operator+=(const char *str) {
  if (str == nullptr) {
    return *this;
  }
  size_t str_size = strlen(str);
  size_t new_size = size_ + str_size;
  char *tmp = reinterpret_cast<char *>(malloc(sizeof(char) * (new_size + 1)));
  if (tmp == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(tmp, this->buffer_, size_ + 1);
  strncat(tmp, str, str_size);
  tmp[new_size] = '\0';
  free(buffer_);
  buffer_ = tmp;
  size_ = new_size;
  return *this;
}

String &String::operator+=(const char ch) {
  char *tmp = reinterpret_cast<char *>(malloc(sizeof(char) * (size_ + 2)));
  if (tmp == nullptr) {
    MS_C_EXCEPTION("malloc data failed");
  }
  memcpy(tmp, this->buffer_, size_ + 1);
  tmp[size_] = ch;
  tmp[size_ + 1] = '\0';
  free(buffer_);
  buffer_ = tmp;
  size_ += 1;
  return *this;
}

String &String::append(size_t count, const char ch) {
  (*this) += ch;
  return *this;
}
String &String::append(const String &str) {
  (*this) += str;
  return *this;
}
String &String::append(const char *str) {
  if (str == nullptr) {
    return *this;
  }
  (*this) += str;
  return *this;
}

int String::compare(const String &str) const { return strcmp(buffer_, str.buffer_); }
int String::compare(const char *str) const { return strcmp(buffer_, str); }

String String::substr(size_t pos, size_t count) const { return String(*this, pos, count); }

String operator+(const String &lhs, const char *rhs) {
  String str = lhs;
  str += rhs;
  return str;
}

String operator+(const char *lhs, const String &rhs) {
  String str = rhs;
  str += lhs;
  return str;
}

bool operator==(const String &lhs, const String &rhs) { return lhs.compare(rhs) == 0; }
bool operator==(const String &lhs, const char *rhs) { return lhs.compare(rhs) == 0; }
bool operator==(const char *lhs, const String &rhs) { return rhs.compare(lhs) == 0; }

String to_String(int32_t value) {
  char tmp[sizeof(int32_t) * 4];
  snprintf(tmp, sizeof(int32_t) * 4, "%d", value);
  return String(tmp, strlen(tmp));
}

String to_String(float value) {
  char tmp[FLT_MAX_10_EXP + 20];
  snprintf(tmp, FLT_MAX_10_EXP + 20, "%f", value);
  return String(tmp, strlen(tmp));
}
}  // namespace mindspore
#endif  // NOT_USE_STL
)RAW";

}  // namespace mindspore::lite::micro
