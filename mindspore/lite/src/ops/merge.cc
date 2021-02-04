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

#include "src/ops/merge.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif
#include "src/tensorlist.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Merge::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Merge;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Merge) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::MergeT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else
int Merge::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Merge();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Merge return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateMerge(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Merge, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *MergeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Merge>(primitive); }
Registry MergeRegistry(schema::PrimitiveType_Merge, MergeCreator);
#endif

InferStatus Merge::AbleToInfer(const std::vector<lite::Tensor *> &inputs) {
  for (auto &input : inputs) {
    if (input->shape().empty()) {
      return HasZeroShape;
    }
    if (input->root_tensor() != nullptr && input->root_tensor()->data_c() != nullptr) {
      continue;
    }
    if (input->data_c() == nullptr) {
      return NotAble;
    }
  }
  return Able;
}

int Merge::Infer(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto *input = inputs[i];
    auto *output = outputs[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "input tensor is nullptr";
      return RET_ERROR;
    }
    if (output == nullptr) {
      MS_LOG(ERROR) << "output tensor is nullptr";
      return RET_ERROR;
    }
    output->set_data_type(input->data_type());
    output->set_shape(input->shape());
    output->set_format(input->format());
    auto data_type = input->data_type();
    if (data_type != kObjectTypeTensorType) {
      continue;
    } else {
      auto input_tensorlist = reinterpret_cast<TensorList *>(input);
      auto output_tensorlist = reinterpret_cast<TensorList *>(output);
      output_tensorlist->set_element_shape(input_tensorlist->element_shape());
      output_tensorlist->set_max_elements_num(input_tensorlist->max_elements_num());
      output_tensorlist->set_tensors_data_type(input_tensorlist->tensors_data_type());
    }
  }
  return RET_OK;
}

int Merge::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(inputs_.size() == 2 * outputs_.size());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->set_data_type(inputs_[i]->data_type());
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<Tensor *> left_part_inputs{};
  left_part_inputs.assign(inputs_.begin(), inputs_.begin() + inputs_.size() / 2);

  std::vector<Tensor *> right_part_inputs{};
  right_part_inputs.assign(inputs_.begin() + inputs_.size() / 2, inputs_.end());

  if (AbleToInfer(left_part_inputs) == Able) {
    return Infer(left_part_inputs, outputs_);
  }

  if (AbleToInfer(right_part_inputs) == Able) {
    return Infer(right_part_inputs, outputs_);
  }

  if (AbleToInfer(left_part_inputs) == HasZeroShape && AbleToInfer(right_part_inputs) == HasZeroShape) {
    return Infer(left_part_inputs, outputs_);
  }

  return RET_INFER_INVALID;
}
}  // namespace lite
}  // namespace mindspore
