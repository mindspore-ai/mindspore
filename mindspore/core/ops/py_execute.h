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

#ifndef MINDSPORE_CORE_OPS_PY_EXECUTE_H_
#define MINDSPORE_CORE_OPS_PY_EXECUTE_H_

#include <vector>
#include <memory>
#include <set>

#include "ops/base_operator.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePyExecute = "PyExecute";
/// \brief Implement for JIT Fallback.
/// Refer to Python API @ref mindspore.ops.PyExecute for more details.
class MIND_API PyExecute : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PyExecute);
  /// \brief Constructor.
  PyExecute() : BaseOperator(kNamePyExecute) { InitIOName({"script", "local_keys", "local_values"}, {"result"}); }
};

class MIND_API PyExecuteInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override;

  std::set<int64_t> GetValueDependArgIndices() const override;

  using InferHandler = abstract::ShapePtr (*)(const std::vector<AbstractBasePtr> &);
  static void set_infer_handler(const InferHandler &infer_handler) { infer_handler_ = infer_handler; }

 private:
  inline static InferHandler infer_handler_{nullptr};
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PY_EXECUTE_H_
