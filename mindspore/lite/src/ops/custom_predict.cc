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
#include "src/ops/custom_predict.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int CustomPredict::GetOutputNum() const { return this->primitive_->value.AsCustomPredict()->outputNum; }
float CustomPredict::GetWeightThreshold() const { return this->primitive_->value.AsCustomPredict()->weightThreshold; }

void CustomPredict::SetOutputNum(int output_num) { this->primitive_->value.AsCustomPredict()->outputNum = output_num; }
void CustomPredict::SetWeightThreshold(float weight_threshold) {
  this->primitive_->value.AsCustomPredict()->weightThreshold = weight_threshold;
}
int CustomPredict::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) { return RET_OK; }
#else
int CustomPredict::GetOutputNum() const { return this->primitive_->value_as_CustomPredict()->outputNum(); }
float CustomPredict::GetWeightThreshold() const {
  return this->primitive_->value_as_CustomPredict()->weightThreshold();
}

int CustomPredict::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_CustomPredict();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "CustomPredict attr is nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateCustomPredict(*fbb, attr->outputNum(), attr->weightThreshold());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_CustomPredict, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *CustomPredictCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<CustomPredict>(primitive);
}
Registry CustomPredictRegistry(schema::PrimitiveType_CustomPredict, CustomPredictCreator);
#endif

int CustomPredict::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.at(0);
  auto output0 = outputs_.at(0);
  auto output1 = outputs_.at(1);
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output0 != nullptr);
  MS_ASSERT(output1 != nullptr);

  std::vector<int> shape;
  shape.push_back(GetOutputNum());

  output0->set_shape(shape);
  output0->set_data_type(kNumberTypeInt32);
  output0->set_format(input->format());
  output1->set_shape(shape);
  output1->set_data_type(kNumberTypeFloat32);
  output1->set_format(input->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
