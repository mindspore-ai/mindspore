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

#include "src/ops/split.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Split::GetNumberSplit() const { return this->primitive_->value.AsSplit()->numberSplit; }
std::vector<int> Split::GetSizeSplit() const { return this->primitive_->value.AsSplit()->sizeSplits; }
int Split::GetSplitDim() const { return this->primitive_->value.AsSplit()->splitDim; }

void Split::SetNumberSplit(int number_split) { this->primitive_->value.AsSplit()->numberSplit = number_split; }
void Split::SetSizeSplits(const std::vector<int> &size_splits) {
  this->primitive_->value.AsSplit()->sizeSplits = size_splits;
}
void Split::SetSplitDim(int split_dim) { this->primitive_->value.AsSplit()->splitDim = split_dim; }

int Split::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Split;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Split) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SplitT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->splitDim = CastToInt(prim.GetAttr("axis")).front();
    attr->numberSplit = CastToInt(prim.GetAttr("output_num")).front();
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

#else

int Split::GetNumberSplit() const { return this->primitive_->value_as_Split()->numberSplit(); }
std::vector<int> Split::GetSizeSplit() const {
  auto fb_vector = this->primitive_->value_as_Split()->sizeSplits();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Split::GetSplitDim() const { return this->primitive_->value_as_Split()->splitDim(); }

int Split::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Split();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Split return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> sizeSplits;
  if (attr->sizeSplits() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->sizeSplits()->size()); i++) {
      sizeSplits.push_back(attr->sizeSplits()->data()[i]);
    }
  }
  auto val_offset = schema::CreateSplitDirect(*fbb, attr->numberSplit(), &sizeSplits, attr->splitDim());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Split, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SplitCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Split>(primitive); }
Registry SplitRegistry(schema::PrimitiveType_Split, SplitCreator);
#endif

namespace {
constexpr int kSplitInputNum = 1;
}  // namespace
int Split::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  if (inputs_.size() < kSplitInputNum) {
    MS_LOG(ERROR) << "inputs number is less to " << kSplitInputNum;
    return RET_ERROR;
  }
  if (outputs_.empty()) {
    MS_LOG(ERROR) << "split has no output.";
    return RET_ERROR;
  }
  for (auto &output : outputs_) {
    output->set_data_type(input->data_type());
    output->set_format(input->format());
  }
  size_splits_ = GetSizeSplit();
  num_split_ = GetNumberSplit() == 0 ? static_cast<int>(outputs_.size()) : GetNumberSplit();
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  size_t split_dim = GetSplitDim() < 0 ? input->shape().size() + GetSplitDim() : GetSplitDim();
  std::vector<int> input_shape = input->shape();
  if (split_dim > input_shape.size()) {
    MS_LOG(ERROR) << "split dim is out of range, which is " << input_shape.size();
    return RET_INPUT_PARAM_INVALID;
  }
  if (static_cast<int>(outputs_.size()) != num_split_) {
    MS_LOG(ERROR) << "outputs number is not equal to " << num_split_;
    return RET_ERROR;
  }
  if (size_splits_.empty()) {
    if (input_shape[split_dim] % num_split_ != 0) {
      MS_LOG(ERROR) << "cannot split to equal size, which dim is " << input_shape[split_dim] << ", num split is "
                    << num_split_;
      return RET_INPUT_PARAM_INVALID;
    }
    for (int i = 0; i < num_split_; ++i) {
      size_splits_.push_back(input_shape[split_dim] / num_split_);
    }
  }
  for (int i = 0; i < num_split_; ++i) {
    std::vector<int> output_shape;
    output_shape.insert(output_shape.begin(), input_shape.begin(), input_shape.end());
    int split_dim_i = input_shape.at(split_dim);
    // support split size is -1 in the end.
    if (i == num_split_ - 1 && size_splits_[i] == -1) {
      for (size_t j = 0; j < size_splits_.size() - 1; ++j) {
        split_dim_i -= size_splits_[j];
      }
      size_splits_[i] = split_dim_i;
    } else {
      split_dim_i = size_splits_[i];
    }
    output_shape.at(split_dim) = split_dim_i;
    outputs_.at(i)->set_shape(output_shape);
    outputs_.at(i)->set_data_type(input->data_type());
    outputs_.at(i)->set_format(input->format());
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
