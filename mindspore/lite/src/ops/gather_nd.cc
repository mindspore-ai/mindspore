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

#include "src/ops/gather_nd.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int GatherNd::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_GatherNd;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_GatherNd) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::GatherNdT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("batchDims") != nullptr) {
      attr->batchDims = static_cast<int32_t>(GetValue<int64_t>(prim.GetAttr("batchDims")));
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int GatherNd::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_GatherNd();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_GatherNd return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateGatherNd(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_GatherNd, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *GatherNdCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<GatherNd>(primitive);
}
Registry GatherNdRegistry(schema::PrimitiveType_GatherNd, GatherNdCreator);
#endif

int GatherNd::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "GatherNd should have two inputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "GatherNd should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto indices = inputs_.at(1);
  MS_ASSERT(indices != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto in_shape = input->shape();
  int in_rank = in_shape.size();
  auto indices_shape = indices->shape();
  int indices_rank = indices_shape.size();
  if (indices_shape.at(indices_rank - 1) > in_rank) {
    MS_LOG(ERROR) << "Input of indices data is error!";
    return RET_ERROR;
  }
  std::vector<int> out_shape;
  int i = 0;
  for (i = 0; i < indices_rank - 1; ++i) {
    out_shape.emplace_back(indices_shape.at(i));
  }
  for (i = indices_shape.at(indices_rank - 1); i < in_rank; ++i) {
    out_shape.emplace_back(in_shape.at(i));
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
