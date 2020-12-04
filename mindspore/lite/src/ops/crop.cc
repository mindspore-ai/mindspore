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

#include "src/ops/crop.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int64_t Crop::GetAxis() const { return this->primitive_->value.AsCrop()->axis; }
std::vector<int64_t> Crop::GetOffsets() const { return this->primitive_->value.AsCrop()->offsets; }

void Crop::SetAxis(int64_t axis) { this->primitive_->value.AsCrop()->axis = axis; }
void Crop::SetOffsets(const std::vector<int64_t> &offsets) { this->primitive_->value.AsCrop()->offsets = offsets; }

#else
int Crop::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Crop();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Crop return nullptr";
    return RET_ERROR;
  }
  std::vector<int64_t> offsets;
  if (attr->offsets() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->offsets()->size()); i++) {
      offsets.push_back(attr->offsets()->data()[i]);
    }
  }
  auto val_offset = schema::CreateCropDirect(*fbb, attr->axis(), &offsets);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Crop, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int64_t Crop::GetAxis() const { return this->primitive_->value_as_Crop()->axis(); }
std::vector<int64_t> Crop::GetOffsets() const {
  auto fb_vector = this->primitive_->value_as_Crop()->offsets();
  return std::vector<int64_t>(fb_vector->begin(), fb_vector->end());
}

PrimitiveC *CropCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Crop>(primitive); }
Registry CropRegistry(schema::PrimitiveType_Crop, CropCreator);
#endif

namespace {
constexpr int kCropOutputNum = 1;
constexpr int kCropInputNum = 2;
}  // namespace
int Crop::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (outputs.size() != kCropOutputNum || inputs.size() != kCropInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return RET_PARAM_INVALID;
  }
  outputs[0]->set_format(inputs[0]->format());
  outputs[0]->set_data_type(inputs[0]->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  outputs[0]->set_shape(inputs[1]->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
