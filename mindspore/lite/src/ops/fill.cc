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

#include "src/ops/fill.h"

#include "src/ops/ops_register.h"
#include "nnacl/fp32/fill.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Fill::GetDims() const { return this->primitive_->value.AsFill()->dims; }

void Fill::SetDims(const std::vector<int> &dims) { this->primitive_->value.AsFill()->dims = dims; }

#else
int Fill::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Fill();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Fill return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> dims;
  if (attr->dims() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->dims()->size()); i++) {
      dims.push_back(attr->dims()->data()[i]);
    }
  }
  auto val_offset = schema::CreateFillDirect(*fbb, &dims);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Fill, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> Fill::GetDims() const {
  auto fb_vector = this->primitive_->value_as_Fill()->dims();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

PrimitiveC *FillCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Fill>(primitive); }
Registry FillRegistry(schema::PrimitiveType_Fill, FillCreator);
#endif

OpParameter *PopulateFillParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param = reinterpret_cast<mindspore::lite::Fill *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  FillParameter *fill_param = reinterpret_cast<FillParameter *>(malloc(sizeof(FillParameter)));
  if (fill_param == nullptr) {
    MS_LOG(ERROR) << "malloc FillParameter failed.";
    return nullptr;
  }
  memset(fill_param, 0, sizeof(FillParameter));
  fill_param->op_parameter_.type_ = primitive->Type();
  auto flatDims = param->GetDims();
  fill_param->num_dims_ = flatDims.size();
  int i = 0;
  for (auto iter = flatDims.begin(); iter != flatDims.end(); iter++) {
    fill_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(fill_param);
}

Registry FillParameterRegistry(schema::PrimitiveType_Fill, PopulateFillParameter);

int Fill::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Fill input or output is null!";
    return RET_ERROR;
  }
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  std::vector<int> output_shape;
  for (size_t i = 0; i < GetDims().size(); i++) {
    output_shape.push_back(GetDims()[i]);
  }
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
