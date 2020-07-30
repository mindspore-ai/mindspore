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
constexpr int kBroadcastToInputNum = 1;
constexpr int kBroadcastToOutputNum = 1;
}  // namespace

int BroadcastTo::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs.size() != kBroadcastToInputNum || outputs.size() != kBroadcastToOutputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size() << ", output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  std::vector<int32_t> dst_shape(this->primitive->value_as_BroadcastTo()->dst_shape()->begin(),
                                 this->primitive->value_as_BroadcastTo()->dst_shape()->end());
  auto input_shape = input->shape();
  std::vector<int> shape(dst_shape.size());
  int input_shape_index = input_shape.size() - 1;
  if (input_shape.size() > dst_shape.size()) {
    MS_LOG(ERROR) << "input shape size " << input_shape.size() << " should <= broadcast to shape size "
                  << dst_shape.size() << "!";
    return RET_PARAM_INVALID;
  }

  for (int i = dst_shape.size() - 1; i >= 0; --i) {
    if (dst_shape[i] < 0) {
      MS_LOG(ERROR) << "shape[" << i << "] = " << dst_shape[i] << " ] should be > 0!";
      return RET_PARAM_INVALID;
    }
    if (input_shape_index >= 0) {
      auto dim = input_shape[input_shape_index];
      if (dim != dst_shape[i] && dim != 1) {
        MS_LOG(ERROR) << "Invalid broadcast shape!";
        return RET_PARAM_INVALID;
      }
    }
    shape[i] = dst_shape[i];
    --input_shape_index;
  }
  outputs[0]->SetFormat(input->GetFormat());
  outputs[0]->set_shape(shape);
  outputs[0]->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
