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

#include "src/ops/strided_slice.h"
#include "src/ops/populate/strided_slice_populate.h"
#include <algorithm>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int StridedSlice::GetBeginMask() const { return this->primitive_->value.AsStridedSlice()->beginMask; }
int StridedSlice::GetEndMask() const { return this->primitive_->value.AsStridedSlice()->endMask; }
int StridedSlice::GetEllipsisMask() const { return this->primitive_->value.AsStridedSlice()->ellipsisMask; }
int StridedSlice::GetNewAxisMask() const { return this->primitive_->value.AsStridedSlice()->newAxisMask; }
int StridedSlice::GetShrinkAxisMask() const { return this->primitive_->value.AsStridedSlice()->shrinkAxisMask; }
std::vector<int> StridedSlice::GetBegin() const { return this->primitive_->value.AsStridedSlice()->begin; }
std::vector<int> StridedSlice::GetEnd() const { return this->primitive_->value.AsStridedSlice()->end; }
std::vector<int> StridedSlice::GetStride() const { return this->primitive_->value.AsStridedSlice()->stride; }
std::vector<int> StridedSlice::GetIsScale() const { return this->primitive_->value.AsStridedSlice()->isScale; }

void StridedSlice::SetBeginMask(int begin_mask) { this->primitive_->value.AsStridedSlice()->beginMask = begin_mask; }
void StridedSlice::SetEndMask(int end_mask) { this->primitive_->value.AsStridedSlice()->endMask = end_mask; }
void StridedSlice::SetEllipsisMask(int ellipsis_mask) {
  this->primitive_->value.AsStridedSlice()->ellipsisMask = ellipsis_mask;
}
void StridedSlice::SetNewAxisMask(int new_axis_mask) {
  this->primitive_->value.AsStridedSlice()->newAxisMask = new_axis_mask;
}
void StridedSlice::SetShrinkAxisMask(int shrink_axis_mask) {
  this->primitive_->value.AsStridedSlice()->shrinkAxisMask = shrink_axis_mask;
}
void StridedSlice::SetBegin(const std::vector<int> &begin) { this->primitive_->value.AsStridedSlice()->begin = begin; }
void StridedSlice::SetEnd(const std::vector<int> &end) { this->primitive_->value.AsStridedSlice()->end = end; }
void StridedSlice::SetStride(const std::vector<int> &stride) {
  this->primitive_->value.AsStridedSlice()->stride = stride;
}
void StridedSlice::SetIsScale(const std::vector<int> &is_scale) {
  this->primitive_->value.AsStridedSlice()->isScale = is_scale;
}

int StridedSlice::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_StridedSlice;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_StridedSlice) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::StridedSliceT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new StridedSlice failed";
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

int StridedSlice::GetBeginMask() const { return this->primitive_->value_as_StridedSlice()->beginMask(); }
int StridedSlice::GetEndMask() const { return this->primitive_->value_as_StridedSlice()->endMask(); }
int StridedSlice::GetEllipsisMask() const { return this->primitive_->value_as_StridedSlice()->ellipsisMask(); }
int StridedSlice::GetNewAxisMask() const { return this->primitive_->value_as_StridedSlice()->newAxisMask(); }
int StridedSlice::GetShrinkAxisMask() const { return this->primitive_->value_as_StridedSlice()->shrinkAxisMask(); }
std::vector<int> StridedSlice::GetBegin() const {
  auto fb_vector = this->primitive_->value_as_StridedSlice()->begin();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSlice::GetEnd() const {
  auto fb_vector = this->primitive_->value_as_StridedSlice()->end();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSlice::GetStride() const {
  auto fb_vector = this->primitive_->value_as_StridedSlice()->stride();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> StridedSlice::GetIsScale() const {
  auto fb_vector = this->primitive_->value_as_StridedSlice()->isScale();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int StridedSlice::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_StridedSlice();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_StridedSlice return nullptr";
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
    schema::CreateStridedSliceDirect(*fbb, attr->beginMask(), attr->endMask(), attr->ellipsisMask(),
                                     attr->newAxisMask(), attr->shrinkAxisMask(), &begin, &end, &stride, &isScale);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_StridedSlice, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *StridedSliceCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<StridedSlice>(primitive);
}
Registry StridedSliceRegistry(schema::PrimitiveType_StridedSlice, StridedSliceCreator);
#endif

namespace {
constexpr size_t kStridedSliceOutputNum = 1;
constexpr size_t kStridedSliceInputNum = 1;
constexpr size_t kStridedSliceMultiInputNumMin = 3;
constexpr size_t kStridedSliceMultiInputNumMax = 5;
}  // namespace
bool StridedSlice::CheckInputs(std::vector<lite::Tensor *> inputs_) {
  for (size_t i = 1; i < inputs_.size(); ++i) {
    if (inputs_.at(i)->data_c() == nullptr) {
      MS_LOG(DEBUG) << "strided_slice has input from other node, which only can be obtained when running.";
      return false;
    }
  }

  return ndim_ <= in_shape_.size();
}

void StridedSlice::ApplyNewAxisMask() {
  for (size_t i = 0; i < new_axis_mask_.size(); i++) {
    if (new_axis_mask_.at(i)) {
      ndim_ += 1;
      in_shape_.insert(in_shape_.begin() + i, 1);
      begins_.at(i) = 0;
      ends_.at(i) = 1;
      strides_.at(i) = 1;

      begins_.emplace_back(0);
      ends_.emplace_back(in_shape_.at(ndim_ - 1));
      strides_.emplace_back(1);

      begins_mask_.at(i) = false;
      ends_mask_.at(i) = false;
      ellipsis_mask_.at(i) = false;
      shrink_axis_mask_.at(i) = false;
    }
  }
}

std::vector<int> StridedSlice::ApplyShrinkMask(std::vector<int> out_shape) {
  auto old_out_shape = out_shape;
  out_shape.clear();
  for (size_t i = 0; i < shrink_axis_mask_.size(); i++) {
    if (shrink_axis_mask_.at(i)) {
      ends_.at(i) = begins_.at(i) + 1;
      strides_.at(i) = 1;
    } else {
      out_shape.emplace_back(old_out_shape.at(i));
    }
  }
  for (size_t i = shrink_axis_mask_.size(); i < old_out_shape.size(); i++) {
    out_shape.emplace_back(old_out_shape.at(i));
  }
  return out_shape;
}

/*only one bit will be used if multiple bits are true.*/
void StridedSlice::ApplyEllipsisMask() {
  for (size_t i = 0; i < ellipsis_mask_.size(); i++) {
    if (ellipsis_mask_.at(i)) {
      begins_.at(i) = 0;
      ends_.at(i) = in_shape_.at(i);
      break;
    }
  }
}

void StridedSlice::ApplyBeginMask() {
  for (size_t i = 0; i < ndim_; i++) {
    if (begins_mask_.at(i)) {
      begins_.at(i) = 0;
    }
  }
}

void StridedSlice::ApplyEndMask() {
  for (size_t i = 0; i < ndim_; i++) {
    if (ends_mask_.at(i)) {
      ends_.at(i) = in_shape_.at(i);
    }
  }
}

void StridedSlice::TransIndexToPositive() {
  for (int i = 0; i < static_cast<int>(begins_.size()); ++i) {
    if (begins_.at(i) < 0) {
      begins_.at(i) += in_shape_.at(i);
    }
    if (ends_.at(i) < 0) {
      ends_.at(i) += in_shape_.at(i);
    }
  }
}

int StridedSlice::HandleAxesInputExist(const std::vector<lite::Tensor *> &inputs) {
  // when axes input exist:
  // input order: data, begin, end, axes(opt), stride(opt)
  auto input_tensor = inputs.at(0);
  MS_ASSERT(input_tensor != nullptr);
  auto begin_tensor = inputs.at(1);
  MS_ASSERT(begin_tensor != nullptr);
  int *begin_data = reinterpret_cast<int *>(begin_tensor->MutableData());
  auto end_tensor = inputs.at(2);
  MS_ASSERT(end_tensor != nullptr);
  int *end_data = reinterpret_cast<int *>(end_tensor->MutableData());
  if (begin_data == nullptr || end_data == nullptr) {
    return RET_INFER_ERR;
  }
  // when input contains axes, begins, ends, strides will be expand to the same length as input rank
  ndim_ = static_cast<int>(input_tensor->shape().size());
  int begin_ndim = begin_tensor->ElementsNum();

  int *axes_data = nullptr;
  auto axes_tensor = inputs.at(3);
  if (axes_tensor->ElementsNum() != 0) {
    MS_ASSERT(axes_tensor->ElementsNum() == begin_ndim);
    axes_data = reinterpret_cast<int *>(axes_tensor->MutableData());
    if (axes_data == nullptr) {
      return RET_INFER_ERR;
    }
  }

  int *stride_data = nullptr;
  auto stride_tensor = inputs.at(4);
  if (stride_tensor->ElementsNum() != 0) {
    MS_ASSERT(stride_tensor->ElementsNum() == begin_ndim);
    stride_data = reinterpret_cast<int *>(stride_tensor->MutableData());
    if (stride_data == nullptr) {
      return RET_INFER_ERR;
    }
  }

  std::vector<int> axes;
  if (axes_data == nullptr) {
    for (int i = 0; i < begin_ndim; ++i) {
      axes.push_back(i);
    }
  } else {
    axes.assign(axes_data, axes_data + begin_ndim);
    for (int i = 0; i < begin_ndim; ++i) {
      if (axes.at(i) < 0) {
        axes.at(i) += ndim_;
      }
    }
  }

  in_shape_.assign(ndim_, 0);
  begins_.assign(ndim_, 0);
  ends_.assign(ndim_, 0);
  strides_.assign(ndim_, 0);
  auto input_shape = input_tensor->shape();
  for (size_t i = 0; i < ndim_; ++i) {
    in_shape_.at(i) = input_shape.at(i);
  }
  for (size_t i = 0; i < ndim_; ++i) {
    auto axes_it = std::find(axes.begin(), axes.end(), i);
    if (axes_it != axes.end()) {
      auto axis = axes_it - axes.begin();
      // begins or ends exceed limit will be set to limit
      begins_.at(i) = std::max(std::min(begin_data[axis], input_shape.at(i) - 1), -input_shape.at(i));
      ends_.at(i) = std::max(std::min(end_data[axis], input_shape.at(i)), -input_shape.at(i) - 1);
      strides_.at(i) = stride_data[axis];
    } else {
      begins_.at(i) = 0;
      ends_.at(i) = input_shape.at(i);
      strides_.at(i) = 1;
    }
  }
  return RET_OK;
}

// note: begin, end, stride length are equal, but may less than rank of input
int StridedSlice::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kStridedSliceOutputNum) {
    MS_LOG(ERROR) << "Invalid output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  if (inputs.size() != kStridedSliceInputNum &&
      !(inputs.size() <= kStridedSliceMultiInputNumMax && inputs.size() >= kStridedSliceMultiInputNumMin)) {
    MS_LOG(ERROR) << "Invalid input size " << inputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  outputs.front()->set_data_type(input->data_type());
  outputs.at(0)->set_format(input->format());
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();
  auto inferflag = infer_flag();
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  in_shape_.clear();
  if (inferflag) {
    in_shape_.assign(input_shape.begin(), input_shape.end());
  }
  begins_.clear();
  ends_.clear();
  strides_.clear();
  if (inputs.size() == kStridedSliceInputNum) {
    ndim_ = static_cast<int>(GetBegin().size());

    for (size_t i = 0; i < ndim_; i++) {
      begins_.emplace_back((GetBegin()).at(i));
      ends_.emplace_back((GetEnd()).at(i));
      strides_.emplace_back((GetStride()).at(i));
    }
  }
  if (!CheckInputs(inputs)) {
    MS_LOG(DEBUG) << "Do infer shape in runtime.";
    return RET_INFER_INVALID;
  }
  if (inputs.size() == 4) {
    // input order: input, begins, ends, strides.
    auto begin_tensor = inputs.at(1);
    int *begin_data = reinterpret_cast<int *>(begin_tensor->MutableData());
    auto end_tensor = inputs.at(2);
    int *end_data = reinterpret_cast<int *>(end_tensor->MutableData());
    auto stride_tensor = inputs.at(3);
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
  }
  if (inputs.size() == 5) {
    // input order: input, begins, end, axes, strides
    auto ret = HandleAxesInputExist(inputs);
    if (ret != RET_OK) {
      return ret;
    }
  }

  // set all mask to original input shape
  begins_mask_.resize(ndim_);
  ends_mask_.resize(ndim_);
  ellipsis_mask_.resize(ndim_);
  new_axis_mask_.resize(ndim_);
  shrink_axis_mask_.resize(ndim_);

  //   convert bit to vector
  for (size_t i = 0; i < ndim_; i++) {
    begins_mask_.at(i) = static_cast<uint32_t>(GetBeginMask()) & (1 << i);
    ends_mask_.at(i) = static_cast<uint32_t>(GetEndMask()) & (1 << i);
    ellipsis_mask_.at(i) = static_cast<uint32_t>(GetEllipsisMask()) & (1 << i);
    new_axis_mask_.at(i) = static_cast<uint32_t>(GetNewAxisMask()) & (1 << i);
    shrink_axis_mask_.at(i) = static_cast<uint32_t>(GetShrinkAxisMask()) & (1 << i);
  }

  ApplyNewAxisMask();
  ApplyBeginMask();
  ApplyEndMask();
  ApplyEllipsisMask();

  if (!inferflag) {
    return RET_OK;
  }
  std::vector<int> output_shape(in_shape_);

  TransIndexToPositive();
  for (size_t i = 0; i < ndim_; i++) {
    if (strides_.at(i) == 0) {
      MS_LOG(ERROR) << "strides should not be 0.";
      return RET_INFER_ERR;
    }
    output_shape.at(i) =
      (ends_.at(i) - begins_.at(i) + strides_.at(i) + (strides_.at(i) < 0 ? 1 : -1)) / strides_.at(i);
  }

  output_shape = ApplyShrinkMask(output_shape);

  outputs.front()->set_shape(output_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
