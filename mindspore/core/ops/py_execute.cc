/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "ops/py_execute.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(PyExecute, BaseOperator);

AbstractBasePtr PyExecuteInfer::InferPy(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(DEBUG) << "item: " << item->ToString();
  }

  if (infer_handler_ == nullptr) {
    MS_LOG(EXCEPTION) << "infer_handler_ should not be null.";
  }
  const auto &abs = infer_handler_(input_args);
  return abs;
}

BaseShapePtr PyExecuteInfer::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  const auto &abs = InferPy(primitive, input_args);
  return abs->BuildShape();
}

TypePtr PyExecuteInfer::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_LOG(EXCEPTION) << "Should not invoke InferType().";
}

AbstractBasePtr PyExecuteInfer::InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return InferPy(primitive, input_args);
}

std::set<int64_t> PyExecuteInfer::GetValueDependArgIndices() const { return {-1}; }

REGISTER_PRIMITIVE_OP_INFER_IMPL(PyExecute, prim::kPrimPyExecute, PyExecuteInfer, false);
}  // namespace ops
}  // namespace mindspore
