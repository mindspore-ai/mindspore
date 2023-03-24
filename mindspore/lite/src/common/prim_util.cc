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
#include <set>
#include <vector>
#include "nnacl/op_base.h"
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
static std::set<schema::PrimitiveType> kTensorListOps = {
  schema::PrimitiveType_TensorListFromTensor, schema::PrimitiveType_TensorListGetItem,
  schema::PrimitiveType_TensorListReserve, schema::PrimitiveType_TensorListSetItem,
  schema::PrimitiveType_TensorListStack};

static const char *const kInnerOpNames[8] = {
  "Inner_ToFormat",    "Inner_GltextureToOpencl",       "Inner_Identity",     "Inner_ShapeFusion",
  "Inner_GraphKernel", "Inner_SplitReduceConcatFusion", "Inner_EncoderLayer", "Inner_DecoderLayer",
};
int GetPrimitiveType(const void *primitive, int schema_version) {
  if (primitive == nullptr) {
    return -1;
  }
  return static_cast<const schema::Primitive *>(primitive)->value_type();
}

const char *GetPrimitiveTypeName(const void *primitive, int schema_version) {
  if (primitive == nullptr) {
    return "NONE";
  }
  return schema::EnumNamePrimitiveType(static_cast<const schema::Primitive *>(primitive)->value_type());
}

const char *PrimitiveCurVersionTypeName(int type) {
  if (type >= static_cast<int>(schema::PrimitiveType_MIN) && type < static_cast<int>(schema::PrimitiveType_MAX)) {
    return schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
  } else if (type >= static_cast<int>(schema::PrimitiveType_MAX)) {
    if (type >= PrimType_InnerOpMin && type < PrimType_InnerOpMax) {
      return kInnerOpNames[type - PrimType_InnerOpMin];
    }
  }
  return "";
}

int GenPrimVersionKey(int primitive_type, int schema_version) { return primitive_type * 1000 + schema_version; }

bool IsPartialNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_PartialFusion;
  }
  return false;
}

bool IsCallNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Call;
  }
  return false;
}

bool IsSwitchNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Switch;
  }
  return false;
}

bool IsSwitchLayerNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_SwitchLayer;
  }
  return false;
}

bool IsCustomNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Custom;
  }
  return false;
}

bool IsTensorListNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    if (kTensorListOps.find(reinterpret_cast<const schema::Primitive *>(primitive)->value_type()) !=
        kTensorListOps.end()) {
      return true;
    }
  }
  return false;
}

int GetPartialGraphIndex(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, -1, "primtive cannot be nullptr");
  int index = -1;
  if (schema_version == SCHEMA_CUR) {
    auto partial_fusion = reinterpret_cast<const schema::Primitive *>(primitive)->value_as_PartialFusion();
    if (partial_fusion == nullptr) {
      return -1;
    }
    index = partial_fusion->sub_graph_index();
  }
  return index;
}
bool IsSharedThreadPoolOp(int op_type) {
  std::vector<schema::PrimitiveType> shared_ops = {mindspore::schema::PrimitiveType_MatMulFusion};
  if (find(shared_ops.begin(), shared_ops.end(), op_type) != shared_ops.end()) {
    return true;
  }
  return false;
}
}  // namespace lite
}  // namespace mindspore
