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

#include "c_ops/broadcast_to.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> BroadcastTo::GetDstShape() const { return this->primitive->value.AsBroadcastTo()->dst_shape; }

void BroadcastTo::SetDstShape(const std::vector<int> &dst_shape) {
  this->primitive->value.AsBroadcastTo()->dst_shape = dst_shape;
}

#else

std::vector<int> BroadcastTo::GetDstShape() const {
  auto fb_vector = this->primitive->value_as_BroadcastTo()->dst_shape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void BroadcastTo::SetDstShape(const std::vector<int> &dst_shape) {}
#endif
namespace {
constexpr int kBroadcastToInputNum = 1;
constexpr int kBroadcastToOutputNum = 1;
}  // namespace

int BroadcastTo::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs.size() != kBroadcastToInputNum || outputs.size() != kBroadcastToOutputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size() << ", output size:" << outputs.size();
    return 1;
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
    return 1;
  }

  for (int i = dst_shape.size() - 1; i >= 0; --i) {
    if (dst_shape[i] < 0) {
      MS_LOG(ERROR) << "shape[" << i << "] = " << dst_shape[i] << " ] should be > 0!";
      return 1;
    }
    if (input_shape_index >= 0) {
      auto dim = input_shape[input_shape_index];
      if (dim != dst_shape[i] && dim != 1) {
        MS_LOG(ERROR) << "Invalid broadcast shape!";
        return 1;
      }
    }
    shape[i] = dst_shape[i];
    --input_shape_index;
  }
  outputs[0]->SetFormat(input->GetFormat());
  outputs[0]->set_shape(shape);
  outputs[0]->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
