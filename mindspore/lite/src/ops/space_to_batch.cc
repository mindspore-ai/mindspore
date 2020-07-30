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

#include <vector>
#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
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
    return RET_PARAM_INVALID;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return RET_PARAM_INVALID;
  }

  auto prim = this->primitive->value_as_SpaceToBatch();
  if (prim->blockShape()->size() != kBlockSizesSize) {
    MS_LOG(ERROR) << "Block shape size should be " << kBlockSizesSize;
    return RET_PARAM_INVALID;
  }
  if (prim->paddings()->size() != kPaddingsSize) {
    MS_LOG(ERROR) << "Crops size should be " << kPaddingsSize;
    return RET_PARAM_INVALID;
  }

  for (auto iter = prim->blockShape()->begin(); iter != prim->blockShape()->end(); ++iter) {
    block_sizes_.emplace_back(*iter);
  }

  in_shape_.clear();
  padded_in_shape_.clear();
  paddings_.clear();
  in_shape_.emplace_back(input_shape.at(kNHWC_n_index));
  padded_in_shape_.emplace_back(input_shape.at(kNHWC_n_index));
  for (int i = 0; i < kBlockSizesSize; i++) {
    in_shape_.emplace_back(input_shape.at(i + 1));
    padded_in_shape_.emplace_back(input_shape.at(i + 1) + (paddings_.at(2 * i) + paddings_.at(2 * i + 1)));
    paddings_.emplace_back(paddings_.at(2 * i));
    paddings_.emplace_back(paddings_.at(2 * i + 1));
    if (paddings_.back() % block_sizes_.at(i)) {
      MS_LOG(ERROR) << "Padded shape does not divide block size " << block_sizes_.at(i);
      return RET_PARAM_INVALID;
    }
  }
  in_shape_.emplace_back(input_shape.at(kNHWC_c_index));
  padded_in_shape_.emplace_back(input_shape.at(kNHWC_c_index));

  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[kNHWC_n_index] =
    input_shape[kNHWC_n_index] * (block_sizes_[kNHWC_n_index] * block_sizes_[kNHWC_h_index]);
  output_shape[kNHWC_h_index] = input_shape[kNHWC_h_index] / block_sizes_[kNHWC_n_index];
  output_shape[kNHWC_w_index] = input_shape[kNHWC_w_index] / block_sizes_[kNHWC_h_index];
  output_shape[kNHWC_c_index] = input_shape[kNHWC_c_index];
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
