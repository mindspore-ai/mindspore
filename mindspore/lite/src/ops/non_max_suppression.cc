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

#include "src/ops/non_max_suppression.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
void NonMaxSuppression::SetCenterPointBox(int centerPointBox) {
  this->primitive_->value.AsNonMaxSuppression()->centerPointBox = centerPointBox;
}

int NonMaxSuppression::GetCenterPointBox() const {
  return this->primitive_->value.AsNonMaxSuppression()->centerPointBox;
}
#else
int NonMaxSuppression::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_NonMaxSuppression();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_NonMaxSuppression return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateNonMaxSuppression(*fbb, attr->centerPointBox());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_NonMaxSuppression, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int NonMaxSuppression::GetCenterPointBox() const {
  return this->primitive_->value_as_NonMaxSuppression()->centerPointBox();
}

PrimitiveC *NonMaxSuppressionCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<NonMaxSuppression>(primitive);
}

Registry NonMaxSuppressionRegistry(schema::PrimitiveType_NonMaxSuppression, NonMaxSuppressionCreator);

#endif
int NonMaxSuppression::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(kNumberTypeInt32);
  output->set_format(input->format());
  MS_LOG(INFO) << "NonMaxSuppression infer shape in runtime.";
  return RET_INFER_INVALID;
}
}  // namespace lite
}  // namespace mindspore
