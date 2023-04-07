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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_PRIMITIVE_TYPE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_PRIMITIVE_TYPE_H_

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
  virtual ~PrimitiveType() = default;

  bool operator==(const std::string &other);
  bool operator!=(const std::string &other);
  bool operator==(mindspore::schema::PrimitiveType other);
  bool operator!=(mindspore::schema::PrimitiveType other);

  PrimitiveType &operator=(const std::string &other);
  PrimitiveType &operator=(const mindspore::schema::PrimitiveType &other);

  std::string PBType() const;
  schema::PrimitiveType FBType() const;

 private:
  std::string protocolbuffers_type_;
  schema::PrimitiveType flatbuffers_type_{schema::PrimitiveType_NONE};
};

inline std::ostream &operator<<(std::ostream &os, const PrimitiveType &type) {
  os << type.PBType();
  return os;
}

inline LogStream &operator<<(LogStream &stream, const PrimitiveType &type) {
  stream << "[PrimitiveType: " << type.PBType() << "]";
  return stream;
}
#endif
}  // namespace mindspore::kernel
#endif
