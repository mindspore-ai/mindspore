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

#include "src/ops/fill.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Fill::GetDims() const { return this->primitive_->value.AsFill()->dims; }

void Fill::SetDims(const std::vector<int> &dims) { this->primitive_->value.AsFill()->dims = dims; }

#else
int Fill::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Fill();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Fill return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> dims;
  if (attr->dims() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->dims()->size()); i++) {
      dims.push_back(attr->dims()->data()[i]);
    }
  }
  auto val_offset = schema::CreateFillDirect(*fbb, &dims);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Fill, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> Fill::GetDims() const {
  auto fb_vector = this->primitive_->value_as_Fill()->dims();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

PrimitiveC *FillCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Fill>(primitive); }
Registry FillRegistry(schema::PrimitiveType_Fill, FillCreator);
#endif

template <typename T>
void CalShape(const T *data, const std::vector<Tensor *> &inputs, std::vector<int> *out_shape, int shape_size) {
  int input_count = inputs[0]->ElementsNum();
  int index = 0;
  int size = 1;
  for (int i = 0; i < shape_size; i++) {
    if (static_cast<int>(data[i]) == -1) {
      index = i;
    } else if (static_cast<int>(data[i]) == 0) {
      size *= inputs[0]->shape().at(i);
    } else {
      size *= data[i];
    }
    out_shape->push_back(data[i]);
  }
  if (static_cast<int>(data[index]) == -1) {
    (*out_shape).at(index) = input_count / size;
  }
}

int Fill::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Fill input or output is null!";
    return RET_ERROR;
  }
  if ((inputs_.size() != kSingleNum && inputs_.size() != kDoubleNum) || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> out_shape;
  if (inputs_.size() == kDoubleNum) {
    auto shape_tensor = inputs_.at(1);
    if (shape_tensor->IsConst()) {
      if (shape_tensor->data_c() == nullptr || (shape_tensor->shape().size() == 1 && shape_tensor->shape()[0] == 0)) {
        MS_LOG(DEBUG) << "reshape to a scalar.";
        output->set_shape(out_shape);
        return RET_OK;
      }
    }
    if (shape_tensor->data_c() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return RET_INFER_INVALID;
    }
    size_t shape_size = shape_tensor->ElementsNum();
    switch (shape_tensor->data_type()) {
      case kNumberTypeInt8: {
        auto data = reinterpret_cast<int8_t *>(shape_tensor->MutableData());
        CalShape<int8_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeInt32: {
        auto data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
        CalShape<int32_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeInt64: {
        auto data = reinterpret_cast<int64_t *>(shape_tensor->MutableData());
        CalShape<int64_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeFloat: {
        auto data = reinterpret_cast<float *>(shape_tensor->MutableData());
        CalShape<float>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeUInt32: {
        auto data = reinterpret_cast<uint32_t *>(shape_tensor->MutableData());
        CalShape<uint32_t>(data, inputs_, &out_shape, shape_size);
      } break;
      default: {
        MS_LOG(ERROR) << "Reshape weight tensor has unsupported dataType: " << shape_tensor->data_type();
        return RET_INFER_ERR;
      }
    }
  } else {
    for (size_t i = 0; i < GetDims().size(); i++) {
      out_shape.push_back(GetDims().at(i));
    }
  }

  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
