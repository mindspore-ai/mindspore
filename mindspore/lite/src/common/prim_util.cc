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

#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "src/common/log_adapter.h"
#include "schema/model_generated.h"
#ifdef ENABLE_V0
#include "schema/model_v0_generated.h"
#endif

namespace mindspore {
namespace lite {
int GetPrimitiveType(const void *primitive) {
  if (primitive == nullptr) {
    return -1;
  }
#ifdef ENABLE_V0
  if (VersionManager::GetInstance()->GetSchemaVersion() == SCHEMA_V0) {
    return static_cast<const schema::v0::Primitive *>(primitive)->value_type();
  }
#endif
  return static_cast<const schema::Primitive *>(primitive)->value_type();
}

const char *PrimitiveTypeName(int type) {
#ifdef ENABLE_V0
  if (VersionManager::GetInstance()->GetSchemaVersion() == SCHEMA_V0) {
    return schema::v0::EnumNamePrimitiveType(static_cast<schema::v0::PrimitiveType>(type));
  }
#endif
  return schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
}

const char *PrimitiveCurVersionTypeName(int type) {
  return schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
}

int GenPrimVersionKey(int primitive_type, int schema_version) { return primitive_type * 1000 + schema_version; }

bool IsPartialNode(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_PartialFusion;
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    return reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_type() ==
           schema::v0::PrimitiveType_Partial;
  }
#endif
  return false;
}

bool IsCallNode(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Call;
  }
  return false;
}

bool IsSwitchNode(const void *primitive) {
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Switch;
  }
  return false;
}

int GetPartialGraphIndex(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int index = -1;
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    auto partial_fusion = reinterpret_cast<const schema::Primitive *>(primitive)->value_as_PartialFusion();
    if (partial_fusion == nullptr) {
      return -1;
    }
    index = partial_fusion->sub_graph_index();
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    auto partial = reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_as_Partial();
    if (partial == nullptr) {
      return -1;
    }
    index = partial->subGraphIndex();
  }
#endif
  return index;
}

bool IsWhileNode(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_While;
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    return reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_type() == schema::v0::PrimitiveType_While;
  }
#endif
  return false;
}

int GetWhileBodySubgraphIndex(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int index = -1;
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    auto while_value = reinterpret_cast<const schema::Primitive *>(primitive)->value_as_While();
    if (while_value == nullptr) {
      return -1;
    }
    index = while_value->body_subgraph_index();
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    auto while_value = reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_as_While();
    if (while_value == nullptr) {
      return -1;
    }
    index = while_value->bodySubgraphIndex();
  }
#endif
  return index;
}

int GetWhileCondSubgraphIndex(const void *primitive) {
  MS_ASSERT(primitive != nullptr);
  int index = -1;
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_CUR) {
    auto while_value = reinterpret_cast<const schema::Primitive *>(primitive)->value_as_While();
    if (while_value == nullptr) {
      return -1;
    }
    index = while_value->cond_subgraph_index();
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    auto while_value = reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_as_While();
    if (while_value == nullptr) {
      return -1;
    }
    index = while_value->condSubgraphIndex();
  }
#endif
  return index;
}
}  // namespace lite
}  // namespace mindspore
