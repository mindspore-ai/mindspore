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

#include "src/ops/expand_dims.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ExpandDims::GetDim() const { return this->primitive_->value.AsExpandDims()->dim; }

void ExpandDims::SetDim(int dim) { this->primitive_->value.AsExpandDims()->dim = dim; }

int ExpandDims::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_ExpandDims;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_ExpandDims) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::ExpandDimsT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    // use axis instead of dim
    if (inputs.at(1)->isa<ValueNode>()) {
      auto axis_tensor = inputs.at(1)->cast<ValueNodePtr>();
      int axis = CastToInt(axis_tensor->value()).front();
      attr->dim = axis;
    } else {
      MS_LOG(ERROR) << "input axis is not value node.";
      delete this->primitive_;
      delete attr;
      this->primitive_ = nullptr;
      attr = nullptr;
      return RET_ERROR;
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else
int ExpandDims::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ExpandDims();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ExpandDims return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateExpandDims(*fbb, attr->dim());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ExpandDims, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int ExpandDims::GetDim() const { return this->primitive_->value_as_ExpandDims()->dim(); }

PrimitiveC *ExpandDimsCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ExpandDims>(primitive);
}
Registry ExpandDimsRegistry(schema::PrimitiveType_ExpandDims, ExpandDimsCreator);
#endif

int ExpandDims::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output size is invalid";
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  int dim = GetDim();
  if (dim < 0) {
    dim += input->shape().size() + 1;
  }
  if (dim > static_cast<int>(input->shape().size())) {
    MS_LOG(ERROR) << "attribute dim out of range";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto out_shape = input->shape();
  out_shape.insert(out_shape.begin() + dim, 1, 1);
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
