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

#include "src/ops/softmax.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int SoftMax::GetAxis() const { return this->primitive_->value.AsSoftMax()->axis; }

int SoftMax::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_SoftMax;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_SoftMax) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SoftMaxT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    auto prim_axis = GetValue<int>(prim.GetAttr("axis"));
    attr->axis = prim_axis;
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void SoftMax::SetAxis(int axis) { this->primitive_->value.AsSoftMax()->axis = axis; }

#else

int SoftMax::GetAxis() const { return this->primitive_->value_as_SoftMax()->axis(); }
int SoftMax::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SoftMax();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SoftMax return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSoftMax(*fbb, attr->axis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SoftMax, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif

int SoftMax::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
