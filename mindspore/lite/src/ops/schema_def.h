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
#ifndef MINDSPORE_LITE_SRC_OPS_SCHEMA_DEF_H_
#define MINDSPORE_LITE_SRC_OPS_SCHEMA_DEF_H_
#include <string>
#include "src/ops/schema_register.h"
#ifdef PRIMITIVE_WRITEABLE
#include "ops/conv2d.h"
#include "schema/inner/model_generated.h"
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF(OP)          \
  namespace mindspore::lite::ops { \
  std::string Gen##OP##Def() {     \
    std::string op_def = "table "; \
    op_def.append(#OP);            \
    op_def.append(" {\n");
#elif PRIMITIVE_WRITEABLE
#define OP_SCHEMA_DEF(OP)                                                   \
  namespace mindspore::lite::ops {                                          \
  mindspore::schema::OP##T *PrimitiveOp2SchemaOp(const mindspore::OP *op) { \
    mindspore::schema::OP##T *result_op = new (std::nothrow) mindspore::schema::OP##T();
#else
#define OP_SCHEMA_DEF(OP)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_ATTR(key, type) op_def.append(#key).append(": ").append(#type).append(";\n");
#elif PRIMITIVE_WRITEABLE
#define OP_ATTR(key, type) result_op->key = op->get_##key();
#else
#define OP_ATTR(key, type)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_ATTR_WITH_VALUE(key, type, value) \
  op_def.append(#key).append(": ").append(#type).append(" = ").append(#value).append(";\n");
#elif PRIMITIVE_WRITEABLE
#define OP_ATTR_WITH_VALUE(key, type, value) result_op->key = op->get_##key();
#else
#define OP_ATTR_WITH_VALUE(key, type, value)
#endif

#ifdef GEN_SCHEMA_DEF
#define OP_SCHEMA_DEF_END(OP)                      \
  op_def.append("}\n\n");                          \
  return op_def;                                   \
  }                                                \
  SchemaOpRegister g_schema_op_##OP(Gen##OP##Def); \
  }  // namespace mindspore::lite::ops
#elif PRIMITIVE_WRITEABLE
#define OP_SCHEMA_DEF_END(OP) \
  return result_op;           \
  }                           \
  }  // namespace mindspore::lite::ops
#else
#define OP_SCHEMA_DEF_END(OP)
#endif
#endif  // MINDSPORE_LITE_SRC_OPS_SCHEMA_DEF_H_
