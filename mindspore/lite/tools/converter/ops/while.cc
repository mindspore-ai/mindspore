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

#include <vector>
#include "tools/converter/ops/while.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

constexpr auto kCondSubgraphIndex = "cond_subgraph_index";
constexpr auto kBodySubgraphIndex = "body_subgraph_index";

namespace mindspore {
namespace lite {
void While::Init(const int64_t cond_subgraph_index, const int64_t body_subgraph_index) {
  this->set_cond_subgraph_index(cond_subgraph_index);
  this->set_body_subgraph_index(body_subgraph_index);
}

void While::set_cond_subgraph_index(const int64_t cond_subgraph_index) {
  this->AddAttr(kCondSubgraphIndex, MakeValue(cond_subgraph_index));
}

int64_t While::get_cond_subgraph_index() const {
  auto value_ptr = this->GetAttr(kCondSubgraphIndex);
  return GetValue<int64_t>(value_ptr);
}

void While::set_body_subgraph_index(const int64_t body_subgraph_index) {
  this->AddAttr(kBodySubgraphIndex, MakeValue(body_subgraph_index));
}

int64_t While::get_body_subgraph_index() const {
  auto value_ptr = this->GetAttr(kBodySubgraphIndex);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr WhileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto While_prim = primitive->cast<PrimWhilePtr>();
  MS_EXCEPTION_IF_NULL(While_prim);
  auto op_name = While_prim->name();
  AbstractBasePtrList output;
  for (int64_t i = 0; i < (int64_t)input_args.size(); i++) {
    auto shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape" + std::to_string(i),
                                                              input_args[i]->BuildShape(), op_name);
    output.push_back(std::make_shared<abstract::AbstractTensor>(input_args[i]->BuildType(), shape));
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace lite
}  // namespace mindspore
