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

#include "src/ops/space_to_batch_nd.h"
#include "src/common/common.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kSpaceToBatchNDOutputNum = 1;
constexpr int kSpaceToBatchNDInputNum = 1;
constexpr int kBlockSizesSize = 2;
constexpr int kPaddingsSize = 4;
}  // namespace

#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SpaceToBatchND::GetBlockShape() const {
  return this->primitive_->value.AsSpaceToBatchND()->blockShape;
}
std::vector<int> SpaceToBatchND::GetPaddings() const { return this->primitive_->value.AsSpaceToBatchND()->paddings; }

void SpaceToBatchND::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive_->value.AsSpaceToBatchND()->blockShape = block_shape;
}
void SpaceToBatchND::SetPaddings(const std::vector<int> &paddings) {
  this->primitive_->value.AsSpaceToBatchND()->paddings = paddings;
}

#else

std::vector<int> SpaceToBatchND::GetBlockShape() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatchND()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SpaceToBatchND::GetPaddings() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatchND()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

#endif  // PRIMITIVE_WRITEABLE

int SpaceToBatchND::InferShape(std::vector<lite::tensor::Tensor *> inputs,
                               std::vector<lite::tensor::Tensor *> outputs) {
  if (outputs.size() != kSpaceToBatchNDOutputNum || inputs.size() != kSpaceToBatchNDInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch_nd only support NHWC now!";
    return RET_ERROR;
  }
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size only support " << kDimension_4d << " now!";
    return RET_ERROR;
  }
  auto block_shape = GetBlockShape();
  if (block_shape.size() != kBlockSizesSize) {
    MS_LOG(ERROR) << "blockShape size != " << kBlockSizesSize;
    return RET_ERROR;
  }
  auto pedding = GetPaddings();
  if (pedding.size() != kPaddingsSize) {
    MS_LOG(ERROR) << "pedding size should be " << kPaddingsSize;
    return RET_ERROR;
  }

  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N] * block_shape[0] * block_shape[1];
  output_shape[NHWC_H] = (input_shape[NHWC_H] + pedding[0] + pedding[1]) / block_shape[0];
  output_shape[NHWC_W] = (input_shape[NHWC_W] + pedding[2] + pedding[3]) / block_shape[1];
  output_shape[NHWC_C] = input_shape[NHWC_C];
  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
