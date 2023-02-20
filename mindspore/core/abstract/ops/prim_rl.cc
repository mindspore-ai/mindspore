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

#include <string>
#include <memory>
#include <vector>

#include "abstract/ops/infer_functions.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace abstract {
constexpr int64_t kMaxElement = 10000;
AbstractBasePtr InferImplTensorArrayStack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &) {
  // Infer TensorArrayStack
  const std::string op_name = primitive->name();
  auto attr_shape = primitive->GetAttr("element_shape");
  if (attr_shape == nullptr) {
    MS_LOG(EXCEPTION) << "No attribute [element_shape] in " << op_name;
  }
  auto attr_dtype = primitive->GetAttr("dtype");
  if (attr_dtype == nullptr) {
    MS_LOG(EXCEPTION) << "No attribute [dtype] in " << op_name;
  }
  auto attr_is_dynamic = primitive->GetAttr("is_dynamic_shape");
  if (attr_is_dynamic == nullptr) {
    MS_LOG(EXCEPTION) << "No attribute [is_dynamic_shape] in " << op_name;
  }
  auto attr_size = primitive->GetAttr("size");
  if (attr_size == nullptr) {
    MS_LOG(EXCEPTION) << "No attribute [size] in " << op_name;
  }
  auto size = GetValue<int64_t>(attr_size);
  auto ele_shape = GetValue<std::vector<int64_t>>(attr_shape);
  auto type = GetValue<TypePtr>(attr_dtype);
  primitive->set_attr("max_element", MakeValue(kMaxElement));
  std::shared_ptr<mindspore::abstract::AbstractTensor> output;

  auto is_dynamic = GetValue<bool>(attr_is_dynamic);
  if (is_dynamic) {
    (void)ele_shape.insert(ele_shape.cbegin(), -1);
  } else {
    (void)ele_shape.insert(ele_shape.cbegin(), size);
  }
  output = std::make_shared<AbstractTensor>(type, std::make_shared<Shape>(ele_shape));

  return output;
}
}  // namespace abstract
}  // namespace mindspore
