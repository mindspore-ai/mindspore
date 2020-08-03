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
constexpr int kSpaceToDepthOutputNum = 1;
constexpr int kSpaceToDepthInputNum = 1;
}

int SpaceToDepth::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kSpaceToDepthOutputNum || inputs.size() != kSpaceToDepthInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return RET_PARAM_INVALID;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return RET_PARAM_INVALID;
  }
  auto prim = this->primitive->value_as_SpaceToDepth();
  int32_t block_size = prim->blockSize();
  if (input_shape[kNHWC_c_index] % (block_size * block_size) != 0 || input_shape[kNHWC_c_index] == 0) {
    MS_LOG(ERROR) << "input dimension c size " << input_shape[kNHWC_c_index] << " should be mulitple of block_size("
                  << block_size << ") * block_size)!";
    return RET_PARAM_INVALID;
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[kNHWC_n_index] = input_shape[kNHWC_n_index];
  output_shape[kNHWC_h_index] = input_shape[kNHWC_h_index] / block_size;
  output_shape[kNHWC_w_index] = input_shape[kNHWC_w_index] / block_size;
  output_shape[kNHWC_c_index] = input_shape[kNHWC_c_index] * (block_size * block_size);
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
