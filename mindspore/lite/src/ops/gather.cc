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

#include "src/ops/gather.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Gather::GetAxis() const { return this->primitive_->value.AsGather()->axis; }
int Gather::GetBatchDims() const { return this->primitive_->value.AsGather()->batchDims; }

void Gather::SetAxis(int axis) { this->primitive_->value.AsGather()->axis = axis; }
void Gather::SetBatchDims(int batch_dims) { this->primitive_->value.AsGather()->batchDims = batch_dims; }
int Gather::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Gather;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Gather) {
    MS_LOG(ERROR) << "Gather primitive value type :  " << schema::EnumNamePrimitiveType(primitive_->value.type)
                  << "is  not equal" << schema::EnumNamePrimitiveType(schema::PrimitiveType_Gather);
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto gather_attr = new (std::nothrow) schema::GatherT();
    if (gather_attr == nullptr) {
      MS_LOG(ERROR) << "new primitive value.value error";
      delete this->primitive_;
      delete gather_attr;
      this->primitive_ = nullptr;
      gather_attr = nullptr;
      return RET_ERROR;
    }
    if (inputs.at(2)->isa<ValueNode>()) {
      ValueNodePtr axis_tensor = inputs.at(2)->cast<ValueNodePtr>();
      int axis = CastToInt(axis_tensor->value()).front();
      gather_attr->axis = axis;
    } else {
      MS_LOG(ERROR) << "input axis is not value node.";
      delete this->primitive_;
      delete gather_attr;
      this->primitive_ = nullptr;
      gather_attr = nullptr;
      return RET_ERROR;
    }
    gather_attr->batchDims = 0;
    this->primitive_->value.value = gather_attr;
  }
  return RET_OK;
}
#else
int Gather::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Gather();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Gather return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateGather(*fbb, attr->axis(), attr->batchDims());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Gather, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Gather::GetAxis() const { return this->primitive_->value_as_Gather()->axis(); }
int Gather::GetBatchDims() const { return this->primitive_->value_as_Gather()->batchDims(); }

PrimitiveC *GatherCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Gather>(primitive); }
Registry GatherRegistry(schema::PrimitiveType_Gather, GatherCreator);
#endif

int Gather::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() < kDoubleNum) {
    MS_LOG(DEBUG) << "Gather should be at least two inputs";
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Gather should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto indices = inputs_.at(1);
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(input != nullptr);
  output->set_data_type(input->data_type());
  if (this->quant_type() == schema::QuantType_WeightQuant) {
    output->set_data_type(kNumberTypeFloat32);
  }
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  int axis = GetAxis();
  int batch_dims = GetBatchDims();
  if (axis < 0) {
    axis += input->shape().size();
  }
  auto indices_shape = indices->shape();
  int indices_rank = indices_shape.size();
  if (batch_dims != 0) {
    MS_LOG(ERROR) << "batchDims  " << batch_dims << " != 0, which is not support";
    return RET_ERROR;
  }
  auto in_shape = input->shape();
  int in_rank = in_shape.size();
  if (in_rank < axis + 1) {
    MS_LOG(ERROR) << "input[0]'s rank is less than axis + 1";
    return RET_ERROR;
  }
  std::vector<int> out_shape{in_shape};
  out_shape.erase(out_shape.begin() + axis);
  for (int i = indices_rank - 1; i >= 0; --i) {
    out_shape.insert(out_shape.begin() + axis, indices_shape.at(i));
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
