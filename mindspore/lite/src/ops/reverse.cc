/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/reverse.h"

#include "src/ops/ops_register.h"
#include "nnacl/fp32/reverse.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Reverse::GetAxis() const { return this->primitive_->value.AsReverse()->axis; }

void Reverse::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsReverse()->axis = axis; }

#else

std::vector<int> Reverse::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_Reverse()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Reverse::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Reverse();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Reverse return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateReverseDirect(*fbb, &axis);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Reverse, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ReverseCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Reverse>(primitive); }
Registry ReverseRegistry(schema::PrimitiveType_Reverse, ReverseCreator);

#endif
OpParameter *PopulateReverseParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto reverse_attr =
    reinterpret_cast<mindspore::lite::Reverse *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ReverseParameter *reverse_param = reinterpret_cast<ReverseParameter *>(malloc(sizeof(ReverseParameter)));
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseParameter failed.";
    return nullptr;
  }
  memset(reverse_param, 0, sizeof(ReverseParameter));
  reverse_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = reverse_attr->GetAxis();
  reverse_param->num_axis_ = flatAxis.size();
  int i = 0;
  for (auto iter = flatAxis.begin(); iter != flatAxis.end(); iter++) {
    reverse_param->axis_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(reverse_param);
}

Registry ReverseParameterRegistry(schema::PrimitiveType_Reverse, PopulateReverseParameter);

}  // namespace lite
}  // namespace mindspore
