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

#include "src/ops/pad.h"
#include <string>
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif
#ifdef SUPPORT_TRAIN
#include <tuple>
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Pad::GetPaddings() const { return this->primitive_->value.AsPad()->paddings; }
int Pad::GetPaddingMode() const { return this->primitive_->value.AsPad()->paddingMode; }
float Pad::GetConstantValue() const { return this->primitive_->value.AsPad()->constantValue; }

void Pad::SetPaddings(const std::vector<int> &paddings) { this->primitive_->value.AsPad()->paddings = paddings; }
void Pad::SetPaddingMode(int padding_mode) {
  this->primitive_->value.AsPad()->paddingMode = (schema::PaddingMode)padding_mode;
}
void Pad::SetConstantValue(float constant_value) { this->primitive_->value.AsPad()->constantValue = constant_value; }
int Pad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Pad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Pad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PadT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    string paddingmode = "REFLECT";
    if (prim.GetAttr("mode") == nullptr) {
#ifdef SUPPORT_TRAIN
      if (prim.name() == "Pad") {
        paddingmode = "CONSTANT";
      } else {
#endif
        MS_LOG(ERROR) << "get mode failed!";
        delete this->primitive_;
        delete attr;
        this->primitive_ = nullptr;
        attr = nullptr;
        return RET_ERROR;
#ifdef SUPPORT_TRAIN
      }
#endif
    } else {
      paddingmode = GetValue<string>(prim.GetAttr("mode"));
    }
    if (paddingmode == "REFLECT") {
      attr->paddingMode = schema::PaddingMode_REFLECT;
    } else if (paddingmode == "SYMMETRIC") {
      attr->paddingMode = schema::PaddingMode_SYMMETRIC;
#ifdef SUPPORT_TRAIN
    } else if (paddingmode == "CONSTANT") {
      attr->paddingMode = schema::PaddingMode_CONSTANT;
      if (prim.GetAttr("paddings") != nullptr) {
        auto paddings = prim.GetAttr("paddings");
        auto str = (*paddings).ToString();
        std::replace(str.begin(), str.end(), ',', ' ');
        std::replace(str.begin(), str.end(), ')', ' ');
        std::replace(str.begin(), str.end(), '(', ' ');
        std::stringstream ss(str);
        for (int i; ss >> i;) {
          attr->paddings.push_back(i);
        }
      }
#endif
    } else {
      MS_LOG(ERROR) << "model type not supported!";
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

std::vector<int> Pad::GetPaddings() const {
  auto fb_vector = this->primitive_->value_as_Pad()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Pad::GetPaddingMode() const { return this->primitive_->value_as_Pad()->paddingMode(); }
float Pad::GetConstantValue() const { return this->primitive_->value_as_Pad()->constantValue(); }

int Pad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Pad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Pad return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> paddings;
  if (attr->paddings() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->paddings()->size()); i++) {
      paddings.push_back(attr->paddings()->data()[i]);
    }
  }
  auto val_offset = schema::CreatePadDirect(*fbb, &paddings, attr->paddingMode(), attr->constantValue());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Pad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PadCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Pad>(primitive); }
Registry PadRegistry(schema::PrimitiveType_Pad, PadCreator);
#endif

int GetPaddingFromInput(const std::vector<Tensor *> &inputs, std::vector<int> *paddings) {
  auto paddings_tensor = inputs.at(1);
  int rank = static_cast<int>(inputs.front()->shape().size());
  MS_ASSERT(paddings_tensor->ElementsNum() == 2 * rank);
  if (paddings_tensor->data_c() == nullptr) {
    return RET_INFER_ERR;
  }
  paddings->clear();
  if (paddings_tensor->data_type() == mindspore::kNumberTypeInt64) {
    auto paddings_data = reinterpret_cast<int64_t *>(paddings_tensor->data_c());
    for (auto i = 0; i < rank; ++i) {
      paddings->emplace_back(paddings_data[i * 2]);
      paddings->emplace_back(paddings_data[i * 2 + 1]);
    }
  } else if (paddings_tensor->data_type() == mindspore::kNumberTypeInt32) {
    auto paddings_data = reinterpret_cast<int32_t *>(paddings_tensor->data_c());
    for (auto i = 0; i < rank; ++i) {
      paddings->emplace_back(paddings_data[i * 2]);
      paddings->emplace_back(paddings_data[i * 2 + 1]);
    }
  }
  return RET_OK;
}

int Pad::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  auto input = inputs.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = outputs.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_format(input->format());
  output->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> paddings;
  if (inputs.size() == 1) {
    paddings = GetPaddings();
  } else {
    GetPaddingFromInput(inputs, &paddings);
  }

  if (paddings.empty()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  std::vector<int> output_shape;
  MS_ASSERT(input->shape().size() <= 4);
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto paddings_index = i;
    auto shape = input_shape.at(i) + paddings.at(2 * paddings_index) + paddings.at(2 * paddings_index + 1);
    output_shape.push_back(shape);
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
