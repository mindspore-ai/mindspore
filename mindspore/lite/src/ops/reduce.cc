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

#include "src/ops/reduce.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Reduce::GetAxes() const { return this->primitive_->value.AsReduce()->axes; }
int Reduce::GetKeepDims() const { return this->primitive_->value.AsReduce()->keepDims; }
int Reduce::GetMode() const { return this->primitive_->value.AsReduce()->mode; }
bool Reduce::GetReduceToEnd() const { return this->primitive_->value.AsReduce()->reduceToEnd; }
float Reduce::GetCoeff() const { return this->primitive_->value.AsReduce()->coeff; }

void Reduce::SetAxes(const std::vector<int> &axes) { this->primitive_->value.AsReduce()->axes = axes; }
void Reduce::SetKeepDims(int keep_dims) { this->primitive_->value.AsReduce()->keepDims = keep_dims; }
void Reduce::SetMode(int mode) { this->primitive_->value.AsReduce()->mode = (schema::ReduceMode)mode; }
void Reduce::SetReduceToEnd(bool reduce_to_end) { this->primitive_->value.AsReduce()->reduceToEnd = reduce_to_end; }
void Reduce::SetCoeff(float coeff) { this->primitive_->value.AsReduce()->coeff = coeff; }

int Reduce::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Reduce;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Reduce) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::ReduceT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.name() == "ReduceMean") {
      attr->mode = schema::ReduceMode_ReduceMean;
    } else if (prim.name() == "ReduceSum") {
      attr->mode = schema::ReduceMode_ReduceSum;
    } else if (prim.name() == "ReduceMax") {
      attr->mode = schema::ReduceMode_ReduceMax;
    } else if (prim.name() == "ReduceMin") {
      attr->mode = schema::ReduceMode_ReduceMin;
    } else if (prim.name() == "ReduceProd") {
      attr->mode = schema::ReduceMode_ReduceProd;
    } else if (prim.name() == "ReduceSumSquare") {
      attr->mode = schema::ReduceMode_ReduceSumSquare;
    } else if (prim.name() == "ReduceAll") {
      attr->mode = schema::ReduceMode_ReduceAll;
    } else {
      MS_LOG(ERROR) << "Not supported reduce mode: " << prim.name();
      return RET_ERROR;
    }

    attr->keepDims = GetValue<bool>(prim.GetAttr("keep_dims"));
    if (inputs.size() == kAnfPopulaterInputNumTwo) {
      auto inputNode = inputs.at(kAnfPopulaterInputNumOne);
      MS_ASSERT(inputNode != nullptr);
      if (inputNode->isa<ValueNode>()) {
        auto valueNode = inputNode->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = (*valTuplPtr)[i];
            MS_ASSERT(elem != nullptr);
            attr->axes.emplace_back(CastToInt(elem).front());
          }
        } else {
          int axes_item = CastToInt(value).front();
          attr->axes.push_back(axes_item);
        }
      }
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

std::vector<int> Reduce::GetAxes() const {
  auto fb_vector = this->primitive_->value_as_Reduce()->axes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Reduce::GetKeepDims() const { return this->primitive_->value_as_Reduce()->keepDims(); }
int Reduce::GetMode() const { return this->primitive_->value_as_Reduce()->mode(); }
bool Reduce::GetReduceToEnd() const { return this->primitive_->value_as_Reduce()->reduceToEnd(); }
float Reduce::GetCoeff() const { return this->primitive_->value_as_Reduce()->coeff(); }
int Reduce::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Reduce();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Reduce return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axes;
  if (attr->axes() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axes()->size()); i++) {
      axes.push_back(attr->axes()->data()[i]);
    }
  }
  auto val_offset =
    schema::CreateReduceDirect(*fbb, &axes, attr->keepDims(), attr->mode(), attr->reduceToEnd(), attr->coeff());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Reduce, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ReduceCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Reduce>(primitive); }
Registry ReduceRegistry(schema::PrimitiveType_Reduce, ReduceCreator);
#endif

namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Reduce::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() < kInputSize || outputs_.size() != kOutputSize) {
    return RET_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  bool keep_dims = static_cast<bool>(GetKeepDims());
  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  const auto &axes = GetAxes();
  auto num_axes = axes.size();
  int rank = static_cast<int>(in_shape.size());
  std::vector<int> actual_axes(axes.begin(), axes.end());

  if (GetReduceToEnd()) {
    if (num_axes != 1) {
      MS_LOG(ERROR) << "Reduce when reduce_to_end, num of axis should be 1, got " << num_axes;
      return RET_ERROR;
    }

    int begin_axis;
    begin_axis = axes.at(0) < 0 ? axes.at(0) + rank : axes.at(0);
    for (auto i = begin_axis + 1; i < rank; ++i) {
      actual_axes.emplace_back(i);
    }
    num_axes = rank - begin_axis;
    keep_dims = false;
  }
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (size_t i = 0; i < in_shape.size(); i++) {
        out_shape.push_back(1);
      }
    }
    output->set_shape(out_shape);
    output->set_data_type(input->data_type());
    return RET_OK;
  }
  // reduce on selected axes
  for (size_t i = 0; i < in_shape.size(); i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if (static_cast<size_t>(actual_axes.at(idx)) == i ||
          static_cast<size_t>(actual_axes.at(idx) + in_shape.size()) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape.at(i));
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
