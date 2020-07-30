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
int Tile::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  auto tile_prim = this->primitive->value_as_Tile();
  MS_ASSERT(tile_prim != nullptr);

  std::vector<int> out_shape;
  std::vector<int> multiples;
  std::copy(tile_prim->multiples()->begin(), tile_prim->multiples()->end(), std::back_inserter(multiples));
  for (size_t i = 0; i < input->shape().size(); ++i) {
    int tmp = input->shape()[i] * multiples[i];
    out_shape.push_back(tmp);
  }

  output->SetFormat(input->GetFormat());
  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
