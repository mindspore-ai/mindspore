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
#include "src/ops/layer_norm.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float LayerNorm::GetEpsilon() const { return this->primitive_->value.AsLayerNorm()->epsilon; }
int LayerNorm::GetBeginNormAxis() const { return this->primitive_->value.AsLayerNorm()->begin_norm_axis; }
int LayerNorm::GetBeginParamsAxis() const { return this->primitive_->value.AsLayerNorm()->begin_params_axis; }

void LayerNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsLayerNorm()->epsilon = epsilon; }
void LayerNorm::SetBeginNormAxis(int axis) { this->primitive_->value.AsLayerNorm()->begin_norm_axis = axis; }
void LayerNorm::SetBeginParamsAxis(int axis) { this->primitive_->value.AsLayerNorm()->begin_params_axis = axis; }

int LayerNorm::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_LayerNorm;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_LayerNorm) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto layer_norm_attr = new (std::nothrow) schema::LayerNormT();
    if (layer_norm_attr == nullptr) {
      MS_LOG(ERROR) << "new primitive value.value error";
      return RET_ERROR;
    }
    auto value_attr = prim.GetAttr("epsilon");
    if (value_attr != nullptr) {
      layer_norm_attr->epsilon = GetValue<float>(value_attr);
    } else {
      layer_norm_attr->epsilon = 1e-7;
    }
    auto norm_axis_attr = prim.GetAttr("begin_norm_axis");
    if (norm_axis_attr != nullptr) {
      layer_norm_attr->begin_norm_axis = GetValue<float>(norm_axis_attr);
    } else {
      layer_norm_attr->begin_norm_axis = -1;
    }
    auto params_axis_attr = prim.GetAttr("begin_params_axis");
    if (params_axis_attr != nullptr) {
      layer_norm_attr->begin_params_axis = GetValue<float>(params_axis_attr);
    } else {
      layer_norm_attr->begin_params_axis = -1;
    }
    this->primitive_->value.value = layer_norm_attr;
  }
  return RET_OK;
}
#else
int LayerNorm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_LayerNorm();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_LayerNorm return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateLayerNorm(*fbb, attr->begin_norm_axis(), attr->begin_params_axis(), attr->epsilon());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LayerNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

float LayerNorm::GetEpsilon() const { return this->primitive_->value_as_LayerNorm()->epsilon(); }
int LayerNorm::GetBeginNormAxis() const { return this->primitive_->value_as_LayerNorm()->begin_norm_axis(); }
int LayerNorm::GetBeginParamsAxis() const { return this->primitive_->value_as_LayerNorm()->begin_params_axis(); }

PrimitiveC *LayerNormCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<LayerNorm>(primitive);
}
Registry LayerNormRegistry(schema::PrimitiveType_LayerNorm, LayerNormCreator);
#endif
int LayerNorm::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (outputs_.size() != kSingleNum || (inputs_.size() != kSingleNum && inputs_.size() != kTripleNum)) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs_.size() << ",input size: " << inputs_.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);
  output->set_format(input->format());
  output->set_data_type(input->data_type());

  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  for (size_t i = GetBeginNormAxis(); i < input_shape.size(); i++) {
    normlized_shape_.push_back(input_shape[i]);
  }
  output->set_shape(input_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
