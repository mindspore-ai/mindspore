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

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

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

int Fill::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Fill input or output is null!";
    return RET_ERROR;
  }
  if ((inputs_.size() != kSingleNum && inputs_.size() != kDoubleNum) || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> output_shape;
  auto param_dims = GetDims();
  for (size_t i = 0; i < param_dims.size(); i++) {
    output_shape.push_back(param_dims.at(i));
  }

  if (inputs_.size() == kDoubleNum) {
    auto input_dims = inputs_.at(1);
    MS_ASSERT(input_dims != nullptr);
    if (input_dims->data_c() == nullptr) {
      return RET_INFER_INVALID;
    }
    int *dims_data = reinterpret_cast<int *>(input_dims->data_c());
    output_shape = std::vector<int>{dims_data, dims_data + input_dims->ElementsNum()};
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
