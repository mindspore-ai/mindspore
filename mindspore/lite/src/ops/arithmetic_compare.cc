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

#include "src/ops/arithmetic_compare.h"

namespace mindspore {
namespace lite {

int ArithmeticCompare::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto res = Arithmetic::InferShape(inputs_, outputs_);
  auto output = outputs_.front();
  output->set_data_type(TypeId::kNumberTypeBool);
  return res;
}
}  // namespace lite
}  // namespace mindspore
