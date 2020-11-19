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
#ifndef MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_TYPE_DEF_H_
#define MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_TYPE_DEF_H_

#include <string>
#include "tools/schema_gen/schema_type_register.h"

#define SCHEMA_ENUM_DEF(T, B)      \
  namespace mindspore::lite::ops { \
  std::string GenEnumDef##T() {    \
    std::string def = "enum ";     \
    def.append(#T);                \
    def.append(" : ");             \
    def.append(#B);                \
    def.append(" {\n");

#define SCHEMA_ENUM_ATTR_WITH_VALUE(key, value) def.append(#key).append(" = ").append(#value).append(",\n");

#define SCHEMA_ENUM_ATTR(key) def.append(#key).append(",\n");

#define OP_SCHEMA_DEF_END(T)                           \
  def.append("}\n\n");                                 \
  return def;                                          \
  }                                                    \
  SchemaTypeRegister g_schema_enum_##T(GenEnumDef##T); \
  }  // namespace mindspore::lite::ops

#endif  // MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_TYPE_DEF_H_
