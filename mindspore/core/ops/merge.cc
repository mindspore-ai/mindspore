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

#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "ops/merge.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr MergeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto inputs_type = input_args[0]->BuildType()->cast<TuplePtr>()->elements();
  auto inputs_shape = input_args[0]->BuildShape()->cast<abstract::TupleShapePtr>()->shape();
  std::map<std::string, TypePtr> args;
  for (size_t i = 0; i != inputs_type.size(); i++) {
    (void)args.insert(std::make_pair("input[" + std::to_string(i) + "]", inputs_type[i]));
  }
  std::set<TypePtr> template_type = common_valid_types;
  (void)template_type.emplace(kBool);
  auto infered_type = CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, template_type, op_name);
  std::vector<int64_t> in_shape0 = inputs_shape[0]->cast<abstract::ShapePtr>()->shape();

  auto output1 = std::make_shared<abstract::AbstractTensor>(infered_type, in_shape0);
  auto output2 = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>{1});

  AbstractBasePtrList output = {output1, output2};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameMerge, Merge);
}  // namespace ops
}  // namespace mindspore
