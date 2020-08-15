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

#include "c_ops/batch_to_space.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> BatchToSpace::GetBlockShape() const { return this->primitive->value.AsBatchToSpace()->blockShape; }
std::vector<int> BatchToSpace::GetCrops() const { return this->primitive->value.AsBatchToSpace()->crops; }

void BatchToSpace::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive->value.AsBatchToSpace()->blockShape = block_shape;
}
void BatchToSpace::SetCrops(const std::vector<int> &crops) { this->primitive->value.AsBatchToSpace()->crops = crops; }

#else

std::vector<int> BatchToSpace::GetBlockShape() const {
  auto fb_vector = this->primitive->value_as_BatchToSpace()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> BatchToSpace::GetCrops() const {
  auto fb_vector = this->primitive->value_as_BatchToSpace()->crops();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void BatchToSpace::SetBlockShape(const std::vector<int> &block_shape) {}
void BatchToSpace::SetCrops(const std::vector<int> &crops) {}
#endif
namespace {
constexpr int kBatchToSpaceOutputNum = 1;
constexpr int kBatchToSpaceInputNum = 1;
constexpr int kBlockShapeSize = 2;
constexpr int kCropsSize = 4;
}  // namespace

int BatchToSpace::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kBatchToSpaceOutputNum || inputs.size() != kBatchToSpaceInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "batch_to_space only support NHWC now!";
    return 1;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return 1;
  }

  auto block_shape = GetBlockShape();
  if (block_shape.size() != kBlockShapeSize) {
    MS_LOG(ERROR) << "Block shape size should be " << kBlockShapeSize;
    return 1;
  }
  auto crops = GetCrops();
  if (crops.size() != kCropsSize) {
    MS_LOG(ERROR) << "Crops size should be " << kCropsSize;
    return 1;
  }
  size_t mul_block_shape = 1;

  for (size_t i = 0; i < kBlockShapeSize; ++i) {
    if (block_shape[i] <= 0) {
      MS_LOG(ERROR) << "Input block_shape should > 0!";
      return 1;
    }
    if (input_shape[NHWC_N] % block_shape[i]) {
      MS_LOG(ERROR) << "Dimension n " << input_shape[NHWC_N] << " can not divide block_shape[" << i << "] "
                    << block_shape[i];
      return 1;
    }
    mul_block_shape *= block_shape[i];
  }

  if (input_shape[NHWC_N] < mul_block_shape) {
    MS_LOG(ERROR) << "Dimension n " << input_shape[NHWC_N] << " < product of block shape!";
    return 1;
  }
  for (size_t i = 0; i < kCropsSize; ++i) {
    if (crops[i] < 0) {
      MS_LOG(ERROR) << "Input crops should >= 0";
      return 1;
    }
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N] / mul_block_shape;
  output_shape[NHWC_H] = input_shape[NHWC_H] * block_shape[0] - crops[0] - crops[1];
  output_shape[NHWC_W] = input_shape[NHWC_W] * block_shape[1] - crops[2] - crops[3];
  output_shape[NHWC_C] = input_shape[NHWC_C];

  outputs[0]->SetFormat(input->GetFormat());
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
