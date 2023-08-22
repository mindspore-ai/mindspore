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

#ifndef MINDSPORE_LITE_SRC_INFER_PRIMITIVE_TYPE_H_
#define MINDSPORE_LITE_SRC_INFER_PRIMITIVE_TYPE_H_

#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#ifdef ENABLE_CLOUD_INFERENCE
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <new>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <map>
#endif

namespace mindspore::kernel {
#ifndef ENABLE_CLOUD_INFERENCE
using PrimitiveType = mindspore::schema::PrimitiveType;
#else
class PrimitiveType {
 public:
  PrimitiveType() = default;
  explicit PrimitiveType(std::string primitive_type);
  explicit PrimitiveType(mindspore::schema::PrimitiveType primitive_type);
  explicit PrimitiveType(int primitive_type);
  virtual ~PrimitiveType() = default;

  bool operator==(const std::string &other) const;
  bool operator!=(const std::string &other) const;
  bool operator==(mindspore::schema::PrimitiveType other) const;
  bool operator!=(mindspore::schema::PrimitiveType other) const;
  bool operator==(int other) const;
  bool operator!=(int other) const;

  PrimitiveType &operator=(const std::string &other);
  PrimitiveType &operator=(const mindspore::schema::PrimitiveType &other);
  PrimitiveType &operator=(int other);

  std::string TypeName() const;
  schema::PrimitiveType SchemaType() const;

 private:
  std::string protocolbuffers_type_;
  int flatbuffers_type_{schema::PrimitiveType_NONE};
};
#endif

inline std::string TypeName(const PrimitiveType &type) {
#ifdef ENABLE_CLOUD_INFERENCE
  return type.TypeName();
#else
  return schema::EnumNamePrimitiveType(type);
#endif
}

inline schema::PrimitiveType SchemaType(const PrimitiveType &type) {
#ifdef ENABLE_CLOUD_INFERENCE
  return type.SchemaType();
#else
  return type;
#endif
}

inline std::ostream &operator<<(std::ostream &os, const PrimitiveType &type) {
  os << TypeName(type);
  return os;
}

#ifdef USE_GLOG
inline LogStream &operator<<(LogStream &stream, const PrimitiveType &type) {
  stream << TypeName(type);
  return stream;
}
#endif
}  // namespace mindspore::kernel
namespace mindspore::lite {
inline bool IsContain(const std::vector<schema::PrimitiveType> &vec, const kernel::PrimitiveType &element) {
  return std::any_of(vec.begin(), vec.end(),
                     [&element](const schema::PrimitiveType &stype) { return element == stype; });
}
}  // namespace mindspore::lite
#endif
