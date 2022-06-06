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

#include "ir/dtype.h"
#include "utils/ms_utils.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "utils/shape_utils.h"

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
  auto is_dynamic = GetValue<bool>(attr_is_dynamic);
  auto size = GetValue<int64_t>(attr_size);
  auto ele_shape = GetValue<std::vector<int64_t>>(attr_shape);
  auto type = GetValue<TypePtr>(attr_dtype);
  primitive->set_attr("max_element", MakeValue(kMaxElement));
  std::shared_ptr<mindspore::abstract::AbstractTensor> output;
  if (is_dynamic) {
    auto max_shape_ = ele_shape;
    auto min_shape_ = ele_shape;
    auto out_shape_ = ele_shape;
    (void)max_shape_.insert(max_shape_.cbegin(), kMaxElement);
    (void)min_shape_.insert(min_shape_.cbegin(), 1);
    (void)out_shape_.insert(out_shape_.cbegin(), -1);
    ShapeVector out_shape = out_shape_;
    ShapeVector min_shape = min_shape_;
    ShapeVector max_shape = max_shape_;
    output = std::make_shared<AbstractTensor>(type, std::make_shared<Shape>(out_shape, min_shape, max_shape));
  } else {
    auto out_shape_ = ele_shape;
    (void)out_shape_.insert(out_shape_.cbegin(), size);
    ShapeVector out_shape = out_shape_;
    output = std::make_shared<AbstractTensor>(type, std::make_shared<Shape>(out_shape));
  }
  return output;
}
}  // namespace abstract
}  // namespace mindspore
