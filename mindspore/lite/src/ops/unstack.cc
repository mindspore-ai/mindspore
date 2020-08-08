/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
int Unstack::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  auto input = inputs.at(0);
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();
  auto prim = this->primitive->value_as_Unstack();
  int axis = prim->axis() < 0 ? prim->axis() + input_shape.size() : prim->axis();
  if (axis < 0 || axis >= input_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis " << prim->axis();
    return RET_PARAM_INVALID;
  }

  std::vector<int> output_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i != axis) {
      output_shape.push_back(input_shape.at(i));
    }
  }
  for (auto &out : outputs) {
    MS_ASSERT(out != nullptr);
    out->set_shape(output_shape);
    out->set_data_type(input->data_type());
    out->SetFormat(input->GetFormat());
  }
  return RET_OK;
}
}  // namespace mindspore::lite
