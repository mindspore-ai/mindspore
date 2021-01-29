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

#include "src/ops/squeeze.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Squeeze::GetAxis() const { return this->primitive_->value.AsSqueeze()->axis; }

void Squeeze::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsSqueeze()->axis = axis; }

int Squeeze::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Squeeze;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Squeeze) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SqueezeT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("axis") == nullptr) {
      MS_LOG(INFO) << "Squeeze's attr xis is set to default";
      attr->axis = {0};
    } else {
      attr->axis = CastToInt(prim.GetAttr("axis"));
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

std::vector<int> Squeeze::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_Squeeze()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Squeeze::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Squeeze();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Squeeze return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateSqueezeDirect(*fbb, &axis);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Squeeze, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SqueezeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Squeeze>(primitive); }
Registry SqueezeRegistry(schema::PrimitiveType_Squeeze, SqueezeCreator);
#endif

namespace {
constexpr int kSqueezeInputNum = 1;
constexpr int kSqueezeOutputNum = 1;
}  // namespace
int Squeeze::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (kSqueezeInputNum != inputs_.size()) {
    MS_LOG(ERROR) << "Add should has " << kSqueezeInputNum << " inputs";
    return -1;
  }
  if (kSqueezeOutputNum != outputs_.size()) {
    MS_LOG(ERROR) << "Add should has " << kSqueezeOutputNum << " outputs";
    return -1;
  }
  auto *in_tensor = inputs_.front();
  outputs_.front()->set_data_type(in_tensor->data_type());
  outputs_.front()->set_format(in_tensor->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto in_shape = in_tensor->shape();
  std::vector<int> out_shape;

  auto axis = GetAxis();
  std::vector<int> axes;
  std::transform(axis.begin(), axis.end(), std::back_inserter(axes),
                 [in_shape](int a) { return a >= 0 ? a : a + in_shape.size(); });
  if (axes.size() == 0) {
    for (size_t i = 0; i < in_shape.size(); i++) {
      if (in_shape.at(i) != 1) {
        out_shape.push_back(in_shape.at(i));
      }
    }
  } else {
    size_t axisIdx = 0;
    for (size_t i = 0; i < in_shape.size(); i++) {
      if (axisIdx < axes.size() && axes.at(axisIdx) == static_cast<int>(i)) {
        MS_ASSERT(in_shape.at(i) == 1);
        axisIdx++;
        continue;
      } else {
        out_shape.push_back(in_shape.at(i));
      }
    }
  }
  outputs_.front()->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
