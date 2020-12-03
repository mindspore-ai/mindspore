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

#include "src/ops/upsample.h"
#include <string>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::string Upsample::GetMode() const { return this->primitive_->value.AsUpsample()->mode; }
std::vector<float> Upsample::GetScales() const { return this->primitive_->value.AsUpsample()->scales; }

void Upsample::SetMode(std::string mode) { this->primitive_->value.AsUpsample()->mode = mode; }
void Upsample::SetScales(const std::vector<float> &scales) { this->primitive_->value.AsUpsample()->scales = scales; }

#else

std::string Upsample::GetMode() const { return this->primitive_->value_as_Upsample()->mode()->str(); }
std::vector<float> Upsample::GetScales() const {
  auto fb_vector = this->primitive_->value_as_Upsample()->scales();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
int Upsample::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Upsample();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Upsample return nullptr";
    return RET_ERROR;
  }
  std::vector<float> scales;
  if (attr->scales() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->scales()->size()); i++) {
      scales.push_back(attr->scales()->data()[i]);
    }
  }
  auto val_offset = schema::CreateUpsampleDirect(*fbb, attr->mode()->c_str(), &scales);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Upsample, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *UpsampleCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Upsample>(primitive);
}
Registry UpsampleRegistry(schema::PrimitiveType_Upsample, UpsampleCreator);

#endif
int Upsample::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  auto input_tensor = inputs_.at(0);
  MS_ASSERT(input_tensor);
  auto input_shape = input_tensor->shape();
  if (input_shape.size() != 4) {
    MS_LOG(ERROR) << "Upsample InferShape input tensor rank should be 4";
    return RET_INFER_ERR;
  }
  auto scale_tensor = inputs_.at(1);
  MS_ASSERT(scale_tensor);
  auto scale_shape = scale_tensor->shape();
  if (scale_shape.size() != 1 && scale_shape.at(0) != 4) {
    MS_LOG(ERROR) << "Upsample scale tensor shape should be 4";
    return RET_INFER_ERR;
  }
  auto scale = reinterpret_cast<float *>(scale_tensor->data_c());
  if (scale == nullptr) {
    MS_LOG(ERROR) << "Upsample scale data nullptr";
    return RET_INFER_INVALID;
  }

  std::vector<int> out_shape = input_shape;  // n, h, w, c; n, c not changed, h = floor(input_h * scale_h).
  int new_height = static_cast<int>(floor(input_shape.at(1) * scale[1]));
  MS_ASSERT(new_height > 0);
  int new_width = static_cast<int>(floor(input_shape.at(2) * scale[2]));
  MS_ASSERT(new_width > 0);
  out_shape.at(1) = new_height;
  out_shape.at(2) = new_width;

  auto out_tensor = outputs_.at(0);
  MS_ASSERT(out_tensor);
  out_tensor->set_shape(out_shape);
  out_tensor->set_data_type(input_tensor->data_type());
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
