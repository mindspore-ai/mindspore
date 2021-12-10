/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BASE_OPERATOR_
#define MINDSPORE_CORE_OPS_BASE_OPERATOR_

#include <string>
#include <memory>
#include <vector>

#include "mindapi/ir/primitive.h"

namespace abstract {
class AnalysisEngine;
using AnalysisEnginePtr = std::shared_ptr<AnalysisEngine>;

class AbstractBase;
using AbstractBasePtr = std::shared_ptr<abstract::AbstractBase>;
}  // namespace abstract

namespace mindspore {
class Primitive;
using PrimitivePtr = std::shared_ptr<Primitive>;
}  // namespace mindspore

namespace mindspore {
namespace ops {
class BaseOperator : public api::Primitive {
 public:
  explicit BaseOperator(const std::string &name);
  ~BaseOperator() = default;

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BASE_OPERATOR_
