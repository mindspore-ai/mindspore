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

#include "src/ops/strided_slice_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE
int StridedSliceGrad::GetBeginMask() const { return this->primitive_->value.AsStridedSliceGrad()->beginMask; }
int StridedSliceGrad::GetEndMask() const { return this->primitive_->value.AsStridedSliceGrad()->endMask; }
int StridedSliceGrad::GetEllipsisMask() const { return this->primitive_->value.AsStridedSliceGrad()->ellipsisMask; }
int StridedSliceGrad::GetNewAxisMask() const { return this->primitive_->value.AsStridedSliceGrad()->newAxisMask; }
int StridedSliceGrad::GetShrinkAxisMask() const { return this->primitive_->value.AsStridedSliceGrad()->shrinkAxisMask; }
std::vector<int> StridedSliceGrad::GetBegin() const { return this->primitive_->value.AsStridedSliceGrad()->begin; }
std::vector<int> StridedSliceGrad::GetEnd() const { return this->primitive_->value.AsStridedSliceGrad()->end; }
std::vector<int> StridedSliceGrad::GetStride() const { return this->primitive_->value.AsStridedSliceGrad()->stride; }
std::vector<int> StridedSliceGrad::GetIsScale() const { return this->primitive_->value.AsStridedSliceGrad()->isScale; }

void StridedSliceGrad::SetBeginMask(int begin_mask) {
  this->primitive_->value.AsStridedSliceGrad()->beginMask = begin_mask;
}
void StridedSliceGrad::SetEndMask(int end_mask) { this->primitive_->value.AsStridedSliceGrad()->endMask = end_mask; }
void StridedSliceGrad::SetEllipsisMask(int ellipsis_mask) {
  this->primitive_->value.AsStridedSliceGrad()->ellipsisMask = ellipsis_mask;
}
void StridedSliceGrad::SetNewAxisMask(int new_axis_mask) {
  this->primitive_->value.AsStridedSliceGrad()->newAxisMask = new_axis_mask;
}
void StridedSliceGrad::SetShrinkAxisMask(int shrink_axis_mask) {
  this->primitive_->value.AsStridedSliceGrad()->shrinkAxisMask = shrink_axis_mask;
}
void StridedSliceGrad::SetBegin(const std::vector<int> &begin) {
  this->primitive_->value.AsStridedSliceGrad()->begin = begin;
}
void StridedSliceGrad::SetEnd(const std::vector<int> &end) { this->primitive_->value.AsStridedSliceGrad()->end = end; }
void StridedSliceGrad::SetStride(const std::vector<int> &stride) {
  this->primitive_->value.AsStridedSliceGrad()->stride = stride;
}
void StridedSliceGrad::SetIsScale(const std::vector<int> &is_scale) {
  this->primitive_->value.AsStridedSliceGrad()->isScale = is_scale;
}

int StridedSliceGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_StridedSliceGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_StridedSliceGrad) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::StridedSliceGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new StridedSliceGrad failed";
      return RET_ERROR;
    }
    attr->beginMask = CastToInt(prim.GetAttr("begin_mask")).front();
    attr->endMask = CastToInt(prim.GetAttr("end_mask")).front();
    attr->ellipsisMask = CastToInt(prim.GetAttr("ellipsis_mask")).front();
    attr->newAxisMask = CastToInt(prim.GetAttr("new_axis_mask")).front();
    attr->shrinkAxisMask = CastToInt(prim.GetAttr("shrink_axis_mask")).front();
    auto inputNodeFirst = inputs[kAnfPopulaterInputNumOne];
    std::vector<int> beginVec;
    GetAttrDataFromInput(inputNodeFirst, &beginVec);
    attr->begin = beginVec;

    auto inputNodeSecond = inputs[kAnfPopulaterInputNumTwo];
    std::vector<int> endVec;
    GetAttrDataFromInput(inputNodeSecond, &endVec);
    attr->end = endVec;

    auto inputNodeThird = inputs[kAnfPopulaterInputNumThree];
    std::vector<int> strideVec;
    GetAttrDataFromInput(inputNodeThird, &strideVec);
    attr->stride = strideVec;
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else

int StridedSliceGrad::GetBeginMask() const { return this->primitive_->value_as_StridedSliceGrad()->beginMask(); }
int StridedSliceGrad::GetEndMask() const { return this->primitive_->value_as_StridedSliceGrad()->endMask(); }
int StridedSliceGrad::GetEllipsisMask() const { return this->primitive_->value_as_StridedSliceGrad()->ellipsisMask(); }
int StridedSliceGrad::GetNewAxisMask() const { return this->primitive_->value_as_StridedSliceGrad()->newAxisMask(); }
int StridedSliceGrad::GetShrinkAxisMask() const {
  return this->primitive_->value_as_StridedSliceGrad()->shrinkAxisMask();
}
std::vector<int> StridedSliceGrad::GetBegin() const {
  auto fb_vector = this->primitive_->value_as_StridedSliceGrad()->begin();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSliceGrad::GetEnd() const {
  auto fb_vector = this->primitive_->value_as_StridedSliceGrad()->end();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSliceGrad::GetStride() const {
  auto fb_vector = this->primitive_->value_as_StridedSliceGrad()->stride();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSliceGrad::GetIsScale() const {
  auto fb_vector = this->primitive_->value_as_StridedSliceGrad()->isScale();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int StridedSliceGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_StridedSliceGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_StridedSliceGrad return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> begin;
  if (attr->begin() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->begin()->size()); i++) {
      begin.push_back(attr->begin()->data()[i]);
    }
  }
  std::vector<int32_t> end;
  if (attr->end() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->end()->size()); i++) {
      end.push_back(attr->end()->data()[i]);
    }
  }
  std::vector<int32_t> stride;
  if (attr->stride() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->stride()->size()); i++) {
      stride.push_back(attr->stride()->data()[i]);
    }
  }
  std::vector<int32_t> isScale;
  if (attr->isScale() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->isScale()->size()); i++) {
      isScale.push_back(attr->isScale()->data()[i]);
    }
  }
  auto val_offset =
    schema::CreateStridedSliceGradDirect(*fbb, attr->beginMask(), attr->endMask(), attr->ellipsisMask(),
                                         attr->newAxisMask(), attr->shrinkAxisMask(), &begin, &end, &stride, &isScale);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_StridedSliceGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *StridedSliceGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<StridedSliceGrad>(primitive);
}
Registry StridedSliceGradRegistry(schema::PrimitiveType_StridedSliceGrad, StridedSliceGradCreator);
#endif

namespace {
constexpr size_t kStridedSliceGradOutputNum = 1;
constexpr size_t kStridedSliceGradMultiInputNumMax = 5;
}  // namespace

int StridedSliceGrad::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kStridedSliceGradOutputNum) {
    MS_LOG(ERROR) << "Invalid output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  if (inputs.size() != kStridedSliceGradMultiInputNumMax) {
    MS_LOG(ERROR) << "Invalid input size " << inputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  outputs.front()->set_data_type(input->data_type());
  outputs.at(0)->set_format(input->format());
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();
  auto inferflag = infer_flag();

  in_shape_.clear();
  if (inferflag) {
    in_shape_.assign(input_shape.begin(), input_shape.end());
  }
  begins_.clear();
  ends_.clear();
  strides_.clear();

  if (!CheckInputs(inputs)) {
    MS_LOG(DEBUG) << "Do infer shape in runtime.";
    return RET_INFER_INVALID;
  }

  // input order: dy, shapex, begins, ends, strides.
  auto begin_tensor = inputs.at(2);
  int *begin_data = reinterpret_cast<int *>(begin_tensor->MutableData());
  auto end_tensor = inputs.at(3);
  int *end_data = reinterpret_cast<int *>(end_tensor->MutableData());
  auto stride_tensor = inputs.at(4);
  int *stride_data = reinterpret_cast<int *>(stride_tensor->MutableData());
  if (begin_data == nullptr || end_data == nullptr || stride_data == nullptr) {
    return RET_INFER_ERR;
  }
  ndim_ = begin_tensor->ElementsNum();
  for (size_t i = 0; i < ndim_; ++i) {
    begins_.emplace_back(begin_data[i]);
    ends_.emplace_back(end_data[i]);
    strides_.emplace_back(stride_data[i]);
  }

  // set all mask to original input shape
  begins_mask_.resize(ndim_);
  ends_mask_.resize(ndim_);
  ellipsis_mask_.resize(ndim_);
  new_axis_mask_.resize(ndim_);
  shrink_axis_mask_.resize(ndim_);

  for (size_t i = 0; i < ndim_; i++) {
    begins_mask_.at(i) = static_cast<bool>(GetBeginMask()) & (1 << i);
    ends_mask_.at(i) = static_cast<bool>(GetEndMask()) & (1 << i);
    ellipsis_mask_.at(i) = static_cast<bool>(GetEllipsisMask()) & (1 << i);
    new_axis_mask_.at(i) = static_cast<bool>(GetNewAxisMask()) & (1 << i);
    shrink_axis_mask_.at(i) = static_cast<bool>(GetShrinkAxisMask()) & (1 << i);
  }

  ApplyNewAxisMask();
  ApplyBeginMask();
  ApplyEndMask();
  ApplyEllipsisMask();

  if (!inferflag) {
    return RET_OK;
  }

  auto output_size = inputs.at(1)->shape().at(0);
  std::vector<int> output_shape;
  MS_ASSERT(inputs.at(1)->MutableData() != nullptr);
  for (int i = 0; i < output_size; i++) {
    output_shape.push_back(static_cast<int *>(inputs.at(1)->MutableData())[i]);
  }
  outputs.front()->set_shape(output_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
