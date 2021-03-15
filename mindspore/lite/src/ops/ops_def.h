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
#ifndef MINDSPORE_LITE_SRC_OPS_OPS_DEF_H_
#define MINDSPORE_LITE_SRC_OPS_OPS_DEF_H_
#include <string>
#include <map>
#include <memory>
#include <utility>
#include "src/ops/ops_func_declare.h"
#include "src/ops/schema_register.h"

#ifdef PRIMITIVE_WRITEABLE
#include "mindspore/core/utils/check_convert_utils.h"
#include "schema/inner/model_generated.h"
#include "schema/inner/ops_types_generated.h"
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_TYPE_DEF_BEGIN(type)        \
  namespace mindspore::lite::ops {     \
  std::string Gen##type() {            \
    std::string prims_type = "union "; \
    prims_type.append(#type).append(" {\n");

#define OP_TYPE(OP) prims_type.append("    ").append(#OP).append(",\n");

#define OP_TYPE_DEF_END(type)                   \
  prims_type.append("}\n");                     \
  return prims_type;                            \
  }                                             \
  PrimitiveTypeRegister g_gen##type(Gen##type); \
  }  // namespace mindspore::lite::ops
#else
#define OP_TYPE_DEF_BEGIN(type)
#define OP_TYPE(OP)
#define OP_TYPE_DEF_END(type)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF(OP)            \
  namespace mindspore::lite::ops {   \
  std::string Gen##OP##Def() {       \
    std::string op_def = "\ntable "; \
    op_def.append(#OP);              \
    op_def.append(" {\n");

#elif PRIMITIVE_WRITEABLE
#define OP_SCHEMA_DEF(OP)                                                      \
  namespace mindspore::lite::ops {                                             \
  mindspore::schema::PrimitiveT *MSOp2SchemaOp(const mindspore::ops::OP *op) { \
    mindspore::schema::OP##T *schema_op = new (std::nothrow) mindspore::schema::OP##T();
#else
#define OP_SCHEMA_DEF(OP)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_ATTR(key, type) op_def.append("    ").append(#key).append(": ").append(#type).append(";\n");
#define OP_ATTR_ENUM(key, type) op_def.append("    ").append(#key).append(": ").append(#type).append(";\n");
#define OP_ATTR_VEC2D(key, type) op_def.append("    ").append(#key).append(": ").append(#type).append(";\n");
#elif PRIMITIVE_WRITEABLE
#define OP_ATTR(key, type)              \
  if (schema_op != nullptr) {           \
    if (op->GetAttr(#key) != nullptr) { \
      schema_op->key = op->get_##key(); \
    }                                   \
  } else {                              \
    return nullptr;                     \
  }

#define OP_ATTR_ENUM(key, type)                                    \
  if (schema_op != nullptr) {                                      \
    if (op->GetAttr(#key) != nullptr) {                            \
      schema_op->key = static_cast<schema::type>(op->get_##key()); \
    }                                                              \
  }

#define OP_ATTR_VEC2D(key, type)                                \
  if (schema_op != nullptr) {                                   \
    auto vec2d = std::make_unique<schema::Vec2DT>();            \
    if (op->GetAttr(#key) != nullptr) {                         \
      auto data = op->get_##key();                              \
      for (size_t i = 0; i < data.size(); ++i) {                \
        auto vec = std::make_unique<schema::VecT>();            \
        vec->data.assign(data.at(i).begin(), data.at(i).end()); \
        vec2d->data.push_back(std::move(vec));                  \
      }                                                         \
      schema_op->key = std::move(vec2d);                        \
    }                                                           \
  }

#else
#define OP_ATTR(key, type)
#define OP_ATTR_ENUM(key, type)
#define OP_ATTR_VEC2D(key, type)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_ATTR_WITH_VALUE(key, type, value) \
  op_def.append("    ").append(#key).append(": ").append(#type).append(" = ").append(#value).append(";\n");
#define OP_ATTR_ENUM_WITH_VALUE(key, type, value) \
  op_def.append("    ").append(#key).append(": ").append(#type).append(" = ").append(#value).append(";\n");
#elif PRIMITIVE_WRITEABLE
#define OP_ATTR_WITH_VALUE(key, type, value) \
  if (schema_op != nullptr) {                \
    if (op->GetAttr(#key) != nullptr) {      \
      schema_op->key = op->get_##key();      \
    }                                        \
  } else {                                   \
    return nullptr;                          \
  }

#define OP_ATTR_ENUM_WITH_VALUE(key, type, value)                  \
  if (schema_op != nullptr) {                                      \
    if (op->GetAttr(#key) != nullptr) {                            \
      schema_op->key = static_cast<schema::type>(op->get_##key()); \
    }                                                              \
  }
#else
#define OP_ATTR_WITH_VALUE(key, type, value)
#define OP_ATTR_ENUM_WITH_VALUE(key, type, value)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF_END(OP)                      \
  op_def.append("}\n");                            \
  return op_def;                                   \
  }                                                \
  SchemaOpRegister g_schema_op_##OP(Gen##OP##Def); \
  }  // namespace mindspore::lite::ops
#elif PRIMITIVE_WRITEABLE
#define OP_SCHEMA_DEF_END(OP)                                         \
  schema::PrimitiveT *prim = new (std::nothrow) schema::PrimitiveT(); \
  if (prim == nullptr) {                                              \
    return nullptr;                                                   \
  }                                                                   \
  prim->value.value = schema_op;                                      \
  prim->value.type = schema::PrimitiveType_##OP;                      \
  return prim;                                                        \
  }                                                                   \
  }  // namespace mindspore::lite::ops
#else
#define OP_SCHEMA_DEF_END(OP)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF_ONLY(OP)       \
  namespace mindspore::lite::ops {   \
  std::string Gen##OP##Def() {       \
    std::string op_def = "\ntable "; \
    op_def.append(#OP);              \
    op_def.append(" {\n");
#else
#define OP_SCHEMA_DEF_ONLY(OP)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_ATTR_ONLY(key, type) op_def.append("    ").append(#key).append(": ").append(#type).append(";\n");
#else
#define OP_ATTR_ONLY(key, type)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF_ONLY_END(OP)                 \
  op_def.append("}\n");                            \
  return op_def;                                   \
  }                                                \
  SchemaOpRegister g_schema_op_##OP(Gen##OP##Def); \
  }  // namespace mindspore::lite::ops
#else
#define OP_SCHEMA_DEF_ONLY_END(OP)
#endif

#endif  // MINDSPORE_LITE_SRC_OPS_OPS_DEF_H_
