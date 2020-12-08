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
#include "src/ops/smooth_l1_loss.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float SmoothL1Loss::GetBeta() const { return this->primitive_->value.AsSmoothL1Loss()->beta; }
int SmoothL1Loss::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_SmoothL1Loss;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_SmoothL1Loss) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = std::make_unique<schema::SmoothL1LossT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->beta = GetValue<float>(prim.GetAttr("beta"));

    this->primitive_->value.value = attr.release();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
float SmoothL1Loss::GetBeta() const { return this->primitive_->value_as_SmoothL1Loss()->beta(); }
int SmoothL1Loss::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SmoothL1Loss();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SmoothL1Loss return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSmoothL1Loss(*fbb, attr->beta());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SmoothL1Loss, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SmoothL1LossCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SmoothL1Loss>(primitive);
}
Registry SmoothL1LossRegistry(schema::PrimitiveType_SmoothL1Loss, SmoothL1LossCreator);
#endif

int SmoothL1Loss::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (inputs.size() != 2) {
    MS_LOG(ERROR) << "SmoothL1Loss should have 2 input tensors";
    return RET_ERROR;
  }

  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "SmoothL1Loss should have 1 output tensors";
    return RET_ERROR;
  }

  if (inputs[0]->ElementsNum() != inputs[1]->ElementsNum()) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }

  auto *out = outputs.front();
  MS_ASSERT(out != nullptr);
  out->set_data_type(inputs[0]->data_type());
  out->set_format(inputs[0]->format());
  out->set_shape(inputs[0]->shape());

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
