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

#include "c_ops/space_to_depth.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int SpaceToDepth::GetBlockSize() const { return this->primitive->value.AsSpaceToDepth()->blockSize; }
int SpaceToDepth::GetFormat() const { return this->primitive->value.AsSpaceToDepth()->format; }

void SpaceToDepth::SetBlockSize(int block_size) { this->primitive->value.AsSpaceToDepth()->blockSize = block_size; }
void SpaceToDepth::SetFormat(int format) { this->primitive->value.AsSpaceToDepth()->format = format; }

#else

int SpaceToDepth::GetBlockSize() const { return this->primitive->value_as_SpaceToDepth()->blockSize(); }
int SpaceToDepth::GetFormat() const { return this->primitive->value_as_SpaceToDepth()->format(); }

void SpaceToDepth::SetBlockSize(int block_size) {}
void SpaceToDepth::SetFormat(int format) {}
#endif
namespace {
constexpr int kSpaceToDepthOutputNum = 1;
constexpr int kSpaceToDepthInputNum = 1;
}  // namespace

int SpaceToDepth::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kSpaceToDepthOutputNum || inputs.size() != kSpaceToDepthInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return 1;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return 1;
  }

  int32_t block_size = GetBlockSize();
  if (input_shape[NHWC_C] % (block_size * block_size) != 0 || input_shape[NHWC_C] == 0) {
    MS_LOG(ERROR) << "input dimension c size " << input_shape[NHWC_C] << " should be mulitple of block_size("
                  << block_size << ") * block_size)!";
    return 1;
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N];
  output_shape[NHWC_H] = input_shape[NHWC_H] / block_size;
  output_shape[NHWC_W] = input_shape[NHWC_W] / block_size;
  output_shape[NHWC_C] = input_shape[NHWC_C] * (block_size * block_size);
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
