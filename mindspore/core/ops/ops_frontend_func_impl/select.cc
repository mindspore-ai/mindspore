/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/select.h"

#include <utility>
#include <memory>
#include <complex>
#include "ops/ops_frontend_func_impl.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;
template <typename T>
void SelectImpl(const bool *conds, void *x, void *y, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(result);
  MS_EXCEPTION_IF_NULL(conds);
  T *x_data = reinterpret_cast<T *>(x);
  T *y_data = reinterpret_cast<T *>(y);
  auto result_data = reinterpret_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(y_data);
  MS_EXCEPTION_IF_NULL(result_data);
  for (size_t i = 0; i < size; ++i) {
    auto cond = conds[i];
    result_data[i] = cond ? x_data[i] : y_data[i];
  }
}

using SelectHandler = std::function<void(const bool *conds, void *x, void *y, void *result, size_t size)>;
std::map<TypeId, SelectHandler> select_impl_list = {
  {kNumberTypeBool, SelectImpl<bool>},
  {kNumberTypeInt8, SelectImpl<int8_t>},
  {kNumberTypeInt16, SelectImpl<int16_t>},
  {kNumberTypeInt32, SelectImpl<int32_t>},
  {kNumberTypeInt64, SelectImpl<int64_t>},
  {kNumberTypeUInt8, SelectImpl<uint8_t>},
  {kNumberTypeUInt16, SelectImpl<uint16_t>},
  {kNumberTypeUInt32, SelectImpl<uint32_t>},
  {kNumberTypeUInt64, SelectImpl<uint64_t>},
  {kNumberTypeFloat16, SelectImpl<float16>},
  {kNumberTypeFloat32, SelectImpl<float>},
  {kNumberTypeFloat64, SelectImpl<double>},
  {kNumberTypeComplex64, SelectImpl<float_complex>},
  {kNumberTypeComplex128, SelectImpl<double_complex>},
};

void SelectInnerInferValue(const PrimitivePtr &prim, const tensor::TensorPtr &cond_tensor,
                           const tensor::TensorPtr &x_tensor, const tensor::TensorPtr &y_tensor,
                           const tensor::TensorPtr &result_tensor) {
  bool *cond_data = reinterpret_cast<bool *>(cond_tensor->data_c());
  auto data_size = cond_tensor->DataSize();
  auto type_id = x_tensor->data_type();
  auto iter = select_impl_list.find(type_id);
  if (iter == select_impl_list.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim->name()
                            << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', "
                               "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', "
                               "'complex128'], but got "
                            << result_tensor->ToString() << ".";
  }

  iter->second(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
  return;
}

class SelectFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto cond_value = input_args[kSelectCondIndex]->GetValue();
    auto x = input_args[kSelectXIndex]->GetValue();
    auto y = input_args[kSelectYIndex]->GetValue();
    if (cond_value == nullptr || x == nullptr || y == nullptr || cond_value->isa<ValueAny>() || x->isa<ValueAny>() ||
        y->isa<ValueAny>()) {
      return nullptr;
    }

    auto x_tensor = x->cast<tensor::TensorPtr>();
    auto y_tensor = y->cast<tensor::TensorPtr>();
    auto cond_tensor = cond_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x_tensor);
    MS_EXCEPTION_IF_NULL(y_tensor);
    MS_EXCEPTION_IF_NULL(cond_tensor);
    auto cond_shape = input_args[kSelectCondIndex]->GetShape()->GetShapeVector();
    auto x_shape = input_args[kSelectXIndex]->GetShape()->GetShapeVector();
    auto y_shape = input_args[kSelectYIndex]->GetShape()->GetShapeVector();
    if (IsDynamic(cond_shape) || IsDynamic(x_shape) || IsDynamic(y_shape)) {
      return nullptr;
    }
    auto conds = cond_tensor->data_c();
    MS_EXCEPTION_IF_NULL(conds);
    auto type_id = x_tensor->data_type();
    auto result_tensor = std::make_shared<tensor::Tensor>(type_id, x_shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    SelectInnerInferValue(primitive, cond_tensor, x_tensor, y_tensor, result_tensor);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Select", SelectFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
