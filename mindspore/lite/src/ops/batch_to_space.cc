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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kBatchToSpaceOutputNum = 1;
constexpr int kBatchToSpaceInputNum = 1;
constexpr int kBlockShapeSize = 2;
constexpr int kCropsSize = 4;
}  // namespace

int BatchToSpace::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kBatchToSpaceOutputNum || inputs.size() != kBatchToSpaceInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return RET_PARAM_INVALID;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "batch_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return RET_PARAM_INVALID;
  }
  auto prim = this->primitive->value_as_BatchToSpace();
  auto block_shape = prim->blockShape();
  if (block_shape->size() != kBlockShapeSize) {
    MS_LOG(ERROR) << "Block shape size should be " << kBlockShapeSize;
    return RET_PARAM_INVALID;
  }
  auto crops = prim->crops();
  if (crops->size() != kCropsSize) {
    MS_LOG(ERROR) << "Crops size should be " << kCropsSize;
    return RET_PARAM_INVALID;
  }
  size_t mul_block_shape = 1;

  for (size_t i = 0; i < kBlockShapeSize; ++i) {
    if (block_shape->Get(i) <= 0) {
      MS_LOG(ERROR) << "Input block_shape should > 0!";
      return RET_PARAM_INVALID;
    }
    if (input_shape[kNHWC_n_index] % block_shape->Get(i)) {
      MS_LOG(ERROR) << "Dimension n " << input_shape[kNHWC_n_index] << " can not divide block_shape[" << i << "] "
                    << block_shape->Get(i);
      return RET_PARAM_INVALID;
    }
    mul_block_shape *= block_shape->Get(i);
  }

  if (input_shape[kNHWC_n_index] < mul_block_shape) {
    MS_LOG(ERROR) << "Dimension n " << input_shape[kNHWC_n_index] << " < product of block shape!";
    return RET_PARAM_INVALID;
  }
  for (size_t i = 0; i < kCropsSize; ++i) {
    if (crops->Get(i) < 0) {
      MS_LOG(ERROR) << "Input crops should >= 0";
      return RET_PARAM_INVALID;
    }
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[kNHWC_n_index] = input_shape[kNHWC_n_index] / mul_block_shape;
  output_shape[kNHWC_h_index] = input_shape[kNHWC_h_index] * block_shape->Get(0) - crops->Get(0) - crops->Get(1);
  output_shape[kNHWC_w_index] = input_shape[kNHWC_w_index] * block_shape->Get(1) - crops->Get(2) - crops->Get(3);
  output_shape[kNHWC_c_index] = input_shape[kNHWC_c_index];

  outputs[0]->SetFormat(input->GetFormat());
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
