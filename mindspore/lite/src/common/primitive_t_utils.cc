/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/common/primitive_t_utils.h"
#ifdef PRIMITIVE_WRITEABLE
#include "src/common/ops/ops_utils.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace lite {
constexpr size_t INITIAL_SIZE = 1024;
const schema::Primitive *ConvertToPrimitive(const schema::PrimitiveT *primitive_t,
                                            flatbuffers::FlatBufferBuilder *fbb) {
  if (primitive_t == nullptr || fbb == nullptr) {
    MS_LOG(ERROR) << "primitiveT or fbb is nullptr.";
    return nullptr;
  }
  auto prim_offset = schema::CreatePrimitive(*fbb, primitive_t);
  fbb->Finish(prim_offset);
  auto prim_buf = fbb->GetBufferPointer();
  return flatbuffers::GetRoot<schema::Primitive>(prim_buf);
}

OpParameter *GetOpParameter(const schema::PrimitiveT *primitive_t) {
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto primitive = ConvertToPrimitive(primitive_t, &fbb);
  fbb.Clear();
  auto prim_type = GetPrimitiveType(primitive, static_cast<int>(SCHEMA_VERSION::SCHEMA_CUR));
  auto parame_gen =
    PopulateRegistry::GetInstance()->GetParameterCreator(prim_type, static_cast<int>(SCHEMA_VERSION::SCHEMA_CUR));
  if (parame_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr.";
    return nullptr;
  }
  auto parameter = parame_gen(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                  << GetPrimitiveTypeName(primitive, static_cast<int>(SCHEMA_VERSION::SCHEMA_CUR));
  }
  return parameter;
}

std::unique_ptr<schema::PrimitiveT> GetPrimitiveT(const std::shared_ptr<mindspore::ops::BaseOperator> &op) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "base operator is nullptr";
    return nullptr;
  }

  if (op->name().empty()) {
    MS_LOG(ERROR) << "the name of operator is null";
    return nullptr;
  }

  auto creator = MSOpsRegistry::GetInstance()->GetPrimitiveCreator(op->name());
  if (creator != nullptr) {
    return creator(op->GetPrim());
  } else {
    MS_LOG(WARNING) << "can not find SingleOpRegistry for operator: " << op->name();
    return nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore
#endif
