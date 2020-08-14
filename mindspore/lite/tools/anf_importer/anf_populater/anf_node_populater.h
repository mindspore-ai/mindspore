/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ANF_NODE_PARSER_H
#define MINDSPORE_ANF_NODE_PARSER_H

#include <vector>
#include "ir/anf.h"
#include "src/ir/primitive_t_value.h"
#include "schema/inner/model_generated.h"
namespace mindspore::lite {
constexpr int kAnfPopulaterOne = 1;
constexpr int kAnfPopulaterTwo = 2;
constexpr int kAnfPopulaterThree = 3;
class AnfNodePopulater {
 public:
  AnfNodePopulater() = default;
  virtual ~AnfNodePopulater() = default;

  virtual int Populate(const PrimitivePtr &prim, PrimitiveTValue *primitiveTValuePtr,
                       const std::vector<AnfNodePtr> &inputs) = 0;
};

}  // namespace mindspore::lite

#endif  // MINDSPORE_ANF_NODE_PARSER_H
