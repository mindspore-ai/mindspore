/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_PRIM_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_PRIM_UTIL_H_

#include "src/common/version_manager.h"
#include "include/api/visible.h"

namespace mindspore {
namespace lite {
MS_API int GetPrimitiveType(const void *primitive, int schema_version = SCHEMA_CUR);
const char *GetPrimitiveTypeName(const void *primitive, int schema_version);
const char *PrimitiveCurVersionTypeName(int type);
MS_API int GenPrimVersionKey(int primitive_type, int schema_version);
bool IsPartialNode(const void *primitive, int schema_version);
bool IsCallNode(const void *primitive, int schema_version);
bool IsSwitchNode(const void *primitive, int schema_version);
bool IsSwitchLayerNode(const void *primitive, int schema_version);
bool IsCustomNode(const void *primitive, int schema_version);
bool IsTensorListNode(const void *primitive, int schema_version);
int GetPartialGraphIndex(const void *primitive, int schema_version);
bool IsSharedThreadPoolOp(int op_type);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_PRIM_UTIL_H_
