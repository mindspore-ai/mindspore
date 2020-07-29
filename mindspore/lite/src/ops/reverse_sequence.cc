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
int ReverseSequence::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  auto input = inputs.front();
  auto output = outputs.front();
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output != nullptr);

  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  return RET_OK;
}
}  // namespace mindspore::lite

