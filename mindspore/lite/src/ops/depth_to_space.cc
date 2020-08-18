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

#include "src/ops/depth_to_space.h"
#include "src/common/common.h"
namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int DepthToSpace::GetBlockSize() const { return this->primitive->value.AsDepthToSpace()->blockSize; }
int DepthToSpace::GetFormat() const { return this->primitive->value.AsDepthToSpace()->format; }

void DepthToSpace::SetBlockSize(int block_size) { this->primitive->value.AsDepthToSpace()->blockSize = block_size; }
void DepthToSpace::SetFormat(int format) { this->primitive->value.AsDepthToSpace()->format = format; }

#else

int DepthToSpace::GetBlockSize() const { return this->primitive->value_as_DepthToSpace()->blockSize(); }
int DepthToSpace::GetFormat() const { return this->primitive->value_as_DepthToSpace()->format(); }

void DepthToSpace::SetBlockSize(int block_size) {}
void DepthToSpace::SetFormat(int format) {}
#endif
namespace {
constexpr int kDepthToSpaceOutputNum = 1;
constexpr int kDepthToSpaceInputNum = 1;
}  // namespace

int DepthToSpace::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kDepthToSpaceOutputNum || inputs.size() != kDepthToSpaceInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "depth_to_space only support NHWC now!";
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
  output_shape[NHWC_H] = input_shape[NHWC_H] * block_size;
  output_shape[NHWC_W] = input_shape[NHWC_W] * block_size;
  output_shape[NHWC_C] = input_shape[NHWC_C] / (block_size * block_size);
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace lite
}  // namespace mindspore
