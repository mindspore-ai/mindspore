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

#include "c_ops/space_to_batch.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SpaceToBatch::GetBlockShape() const { return this->primitive->value.AsSpaceToBatch()->blockShape; }
std::vector<int> SpaceToBatch::GetPaddings() const { return this->primitive->value.AsSpaceToBatch()->paddings; }

void SpaceToBatch::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive->value.AsSpaceToBatch()->blockShape = block_shape;
}
void SpaceToBatch::SetPaddings(const std::vector<int> &paddings) {
  this->primitive->value.AsSpaceToBatch()->paddings = paddings;
}

#else

std::vector<int> SpaceToBatch::GetBlockShape() const {
  auto fb_vector = this->primitive->value_as_SpaceToBatch()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SpaceToBatch::GetPaddings() const {
  auto fb_vector = this->primitive->value_as_SpaceToBatch()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void SpaceToBatch::SetBlockShape(const std::vector<int> &block_shape) {}
void SpaceToBatch::SetPaddings(const std::vector<int> &paddings) {}
#endif
namespace {
constexpr int kSpaceToBatchNDOutputNum = 1;
constexpr int kSpaceToBatchNDInputNum = 1;
constexpr int kBlockSizesSize = 2;
constexpr int kPaddingsSize = 4;
}  // namespace

int SpaceToBatch::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kSpaceToBatchNDOutputNum || inputs.size() != kSpaceToBatchNDInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return 1;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return 1;
  }

  if (GetBlockShape().size() != kBlockSizesSize) {
    MS_LOG(ERROR) << "Block shape size should be " << kBlockSizesSize;
    return 1;
  }
  if (GetPaddings().size() != kPaddingsSize) {
    MS_LOG(ERROR) << "Crops size should be " << kPaddingsSize;
    return 1;
  }

  for (int &iter : GetBlockShape()) {
    block_sizes_.emplace_back(iter);
  }

  in_shape_.clear();
  padded_in_shape_.clear();
  paddings_.clear();
  in_shape_.emplace_back(input_shape.at(NHWC_N));
  padded_in_shape_.emplace_back(input_shape.at(NHWC_N));
  for (int i = 0; i < kBlockSizesSize; i++) {
    in_shape_.emplace_back(input_shape.at(i + 1));
    padded_in_shape_.emplace_back(input_shape.at(i + 1) + (paddings_.at(2 * i) + paddings_.at(2 * i + 1)));
    paddings_.emplace_back(paddings_.at(2 * i));
    paddings_.emplace_back(paddings_.at(2 * i + 1));
    if (paddings_.back() % block_sizes_.at(i)) {
      MS_LOG(ERROR) << "Padded shape does not divide block size " << block_sizes_.at(i);
      return 1;
    }
  }
  in_shape_.emplace_back(input_shape.at(NHWC_C));
  padded_in_shape_.emplace_back(input_shape.at(NHWC_C));

  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N] * (block_sizes_[NHWC_N] * block_sizes_[NHWC_H]);
  output_shape[NHWC_H] = input_shape[NHWC_H] / block_sizes_[NHWC_N];
  output_shape[NHWC_W] = input_shape[NHWC_W] / block_sizes_[NHWC_H];
  output_shape[NHWC_C] = input_shape[NHWC_C];
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
