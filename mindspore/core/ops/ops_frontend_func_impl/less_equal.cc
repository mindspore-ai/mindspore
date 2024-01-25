/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <functional>
#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
template <typename T>
void ImplLessEqual(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = std::less_equal<T>()(x1_data[i], x2_data[i]);
  }
}

using Handler = std::function<void(void *x1, void *x2, void *result, size_t size)>;
std::map<TypeId, Handler> less_equal_impl_list = {
  {kNumberTypeBool, ImplLessEqual<bool>},        {kNumberTypeInt8, ImplLessEqual<int8_t>},
  {kNumberTypeInt16, ImplLessEqual<int16_t>},    {kNumberTypeInt32, ImplLessEqual<int32_t>},
  {kNumberTypeInt64, ImplLessEqual<int64_t>},    {kNumberTypeUInt8, ImplLessEqual<uint8_t>},
  {kNumberTypeUInt16, ImplLessEqual<uint16_t>},  {kNumberTypeUInt32, ImplLessEqual<uint32_t>},
  {kNumberTypeUInt64, ImplLessEqual<uint64_t>},  {kNumberTypeFloat16, ImplLessEqual<float16>},
  {kNumberTypeFloat32, ImplLessEqual<float>},    {kNumberTypeFloat64, ImplLessEqual<double>},
  {kNumberTypeBFloat16, ImplLessEqual<bfloat16>}};

class LessEqualFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x1 = input_args[kIndex0]->GetValue();
    auto x2 = input_args[kIndex1]->GetValue();
    if (x1 == nullptr || x2 == nullptr || x1->isa<ValueAny>() || x2->isa<ValueAny>()) {
      return nullptr;
    }
    auto x1_tensor = x1->cast<tensor::TensorPtr>();
    auto x2_tensor = x2->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_tensor);
    MS_EXCEPTION_IF_NULL(x2_tensor);

    auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto x2_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
    if (IsDynamic(x1_shape) || IsDynamic(x2_shape) || !IsMactchedShapeInferValue(x1_shape, x2_shape)) {
      return nullptr;
    }
    auto type_id = x1_tensor->data_type();
    auto data_size = x1_tensor->DataSize();
    auto result_tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, x1_shape);
    auto iter = less_equal_impl_list.find(type_id);
    if (iter == less_equal_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                    << ", the type is not supported.";
      return nullptr;
    }
    iter->second(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("LessEqual", LessEqualFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
