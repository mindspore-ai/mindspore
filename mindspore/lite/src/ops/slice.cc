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

#include "src/ops/slice.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kSliceInputNum = 1;
constexpr int kSliceOutputNum = 1;
}  // namespace
#ifdef PRIMITIVE_WRITEABLE
int Slice::GetFormat() const { return this->primitive_->value.AsSlice()->format; }
std::vector<int> Slice::GetBegin() const { return this->primitive_->value.AsSlice()->begin; }
std::vector<int> Slice::GetSize() const { return this->primitive_->value.AsSlice()->size; }
std::vector<int> Slice::GetAxes() const { return this->primitive_->value.AsSlice()->axes; }

void Slice::SetFormat(int format) { this->primitive_->value.AsSlice()->format = (schema::Format)format; }
void Slice::SetBegin(const std::vector<int> &begin) { this->primitive_->value.AsSlice()->begin = begin; }
void Slice::SetSize(const std::vector<int> &size) { this->primitive_->value.AsSlice()->size = size; }

#else

int Slice::GetFormat() const { return this->primitive_->value_as_Slice()->format(); }
std::vector<int> Slice::GetBegin() const {
  auto fb_vector = this->primitive_->value_as_Slice()->begin();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> Slice::GetSize() const {
  auto fb_vector = this->primitive_->value_as_Slice()->size();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> Slice::GetAxes() const {
  auto fb_vector = this->primitive_->value_as_Slice()->axes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Slice::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_Slice();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Slice return nullptr";
    return RET_ERROR;
  }

  std::vector<int32_t> axes;
  if (attr->axes() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axes()->size()); i++) {
      axes.push_back(attr->axes()->data()[i]);
    }
  }
  std::vector<int32_t> begin;
  if (attr->begin() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->begin()->size()); i++) {
      begin.push_back(attr->begin()->data()[i]);
    }
  }
  std::vector<int32_t> size;
  if (attr->size() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->size()->size()); i++) {
      size.push_back(attr->size()->data()[i]);
    }
  }

  auto val_offset = schema::CreateSliceDirect(*fbb, attr->format(), &axes, &begin, &size);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Slice, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif

std::vector<int> Slice::GetPostProcessBegin() const { return this->begin; }
std::vector<int> Slice::GetPostProcessSize() const { return this->size; }
int Slice::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs.size() != kSliceInputNum || outputs.size() != kSliceOutputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size() << ",output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();
  std::vector<int32_t> slice_begin(GetBegin());
  std::vector<int32_t> slice_size(GetSize());
  std::vector<int32_t> slice_axes(GetAxes());
  std::vector<int32_t> output_shape(input_shape.size());
  begin.assign(input_shape.size(), 0);
  size.assign(input_shape.size(), -1);
  for (size_t i = 0; i < slice_axes.size(); ++i) {
    begin[slice_axes[i]] = slice_begin[i];
    size[slice_axes[i]] = slice_size[i];
  }
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (size[i] < 0 && size[i] != -1) {
      MS_LOG(ERROR) << "Invalid size input!size[" << i << "]=" << size[i];
      return RET_PARAM_INVALID;
    }
    if (begin[i] < 0) {
      MS_LOG(ERROR) << "Invalid begin input " << begin[i] << " which should be >= 0";
      return RET_PARAM_INVALID;
    }
    if (input_shape[i] <= begin[i]) {
      MS_LOG(ERROR) << "Invalid begin input!begin[" << i << "]=" << begin[i]
                    << " which should be <= " << input_shape[i];
      return RET_PARAM_INVALID;
    }
    if (size[i] > (input_shape[i] - begin[i])) {
      MS_LOG(ERROR) << "Invalid size input " << size[i] << " which should be <= " << input_shape[i] - begin[i];
      return RET_PARAM_INVALID;
    }

    output_shape[i] = size[i] < 0 ? input_shape[i] - begin[i] : size[i];
  }

  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
