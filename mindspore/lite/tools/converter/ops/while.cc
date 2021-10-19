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
#include "tools/common/tensor_util.h"
#include "tools/converter/ops/while.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "nnacl/op_base.h"

constexpr auto kCondSubgraphIndex = "cond_subgraph_index";
constexpr auto kBodySubgraphIndex = "body_subgraph_index";

namespace mindspore {
namespace lite {
void While::Init(const int64_t cond_subgraph_index, const int64_t body_subgraph_index) {
  this->set_cond_subgraph_index(cond_subgraph_index);
  this->set_body_subgraph_index(body_subgraph_index);
}

void While::set_cond_subgraph_index(const int64_t cond_subgraph_index) {
  auto value_ptr = MakeValue(cond_subgraph_index);
  if (value_ptr == nullptr) {
    MS_LOG(ERROR) << "value_ptr is nullptr.";
    return;
  }
  this->AddAttr(kCondSubgraphIndex, value_ptr);
}

int64_t While::get_cond_subgraph_index() const {
  auto value_ptr = this->GetAttr(kCondSubgraphIndex);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, -1);
  return GetValue<int64_t>(value_ptr);
}

void While::set_body_subgraph_index(const int64_t body_subgraph_index) {
  auto value_ptr = MakeValue(body_subgraph_index);
  if (value_ptr == nullptr) {
    MS_LOG(ERROR) << "value_ptr is nullptr.";
    return;
  }
  this->AddAttr(kBodySubgraphIndex, value_ptr);
}

int64_t While::get_body_subgraph_index() const {
  auto value_ptr = this->GetAttr(kBodySubgraphIndex);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, -1);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr WhileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  auto while_prim = primitive->cast<PrimWhilePtr>();
  MS_CHECK_TRUE_RET(while_prim != nullptr, nullptr);
  AbstractBasePtrList output;
  for (size_t i = 0; i < input_args.size(); i++) {
    auto build_shape_ptr = input_args[i]->BuildShape();
    MS_CHECK_TRUE_RET(build_shape_ptr != nullptr, nullptr);
    auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(build_shape_ptr)[kShape];
    auto build_type_ptr = input_args[i]->BuildType();
    MS_CHECK_TRUE_RET(build_type_ptr != nullptr, nullptr);
    auto abstract_tensor = lite::CreateTensorAbstract(shape, build_type_ptr->type_id());
    MS_CHECK_TRUE_RET(abstract_tensor != nullptr, nullptr);
    output.push_back(abstract_tensor);
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace lite
}  // namespace mindspore
