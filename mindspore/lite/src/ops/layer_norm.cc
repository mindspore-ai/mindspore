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
std::vector<int> LayerNorm::GetNormalizedShape() const {
  return this->primitive_->value.AsLayerNorm()->normalizedShape;
}
float LayerNorm::GetEpsilon() const { return this->primitive_->value.AsLayerNorm()->epsilon; }
bool LayerNorm::GetElementwiseAffine() const { return this->primitive_->value.AsLayerNorm()->elementwiseAffine; }

void LayerNorm::SetNormalizedShape(const std::vector<int> &normalizedShape) {
  this->primitive_->value.AsLayerNorm()->normalizedShape = normalizedShape;
}
void LayerNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsLayerNorm()->epsilon = epsilon; }
void LayerNorm::SetElementwiseAffine(bool elementwiseAffine) {
  this->primitive_->value.AsLayerNorm()->elementwiseAffine = elementwiseAffine;
}
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
    value_attr = prim.GetAttr("normalized_shape");
    if (value_attr != nullptr) {
      layer_norm_attr->normalizedShape = CastToInt(value_attr);
    }
    if (inputs.size() == 3) {
      layer_norm_attr->elementwiseAffine = true;
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

  std::vector<int32_t> normalizedShape;
  if (attr->normalizedShape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->normalizedShape()->size()); i++) {
      normalizedShape.push_back(attr->normalizedShape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateLayerNormDirect(*fbb, &normalizedShape, attr->epsilon(), attr->elementwiseAffine());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LayerNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> LayerNorm::GetNormalizedShape() const {
  auto fb_vector = this->primitive_->value_as_LayerNorm()->normalizedShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
float LayerNorm::GetEpsilon() const { return this->primitive_->value_as_LayerNorm()->epsilon(); }
bool LayerNorm::GetElementwiseAffine() const { return this->primitive_->value_as_LayerNorm()->elementwiseAffine(); }
PrimitiveC *LayerNormCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<LayerNorm>(primitive);
}
Registry LayerNormRegistry(schema::PrimitiveType_LayerNorm, LayerNormCreator);

#endif
int LayerNorm::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (outputs_.size() != kSingleNum || (inputs_.size() != kSingleNum && inputs_.size() != kMultiNum)) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs_.size() << ",input size: " << inputs_.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);
  output->set_format(input->format());
  output->set_data_type(input->data_type());

  if (GetElementwiseAffine() && inputs_.size() != kMultiNum) {
    MS_LOG(INFO) << "input tensor amount error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (!GetElementwiseAffine() && inputs_.size() != kSingleNum) {
    MS_LOG(INFO) << "input tensor amount error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  normlized_shape_ = GetNormalizedShape();
  elementwise_mode_ = GetElementwiseAffine() ? 2 : 0;
  if (normlized_shape_.size() > input_shape.size()) {
    MS_LOG(INFO) << "normalized_shape attr invalid";
    return RET_PARAM_INVALID;
  }
  if (normlized_shape_.empty()) {
    // instance norm -> layernorm only for nchw
    if (input->format() == schema::Format_NCHW) {
      normlized_shape_.insert(normlized_shape_.begin(), input_shape.begin() + 2, input_shape.end());
      elementwise_mode_ = 1;
    } else {
      normlized_shape_.insert(normlized_shape_.begin(), input_shape.begin() + 1, input_shape.end());
    }
  }
  size_t first_index = input_shape.size() - normlized_shape_.size();
  for (size_t i = first_index; i < input_shape.size(); ++i) {
    if (input_shape.at(i) != normlized_shape_.at(i - first_index)) {
      MS_LOG(INFO) << "normalized_shape attr invalid";
      return RET_PARAM_INVALID;
    }
  }

  output->set_shape(input_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
