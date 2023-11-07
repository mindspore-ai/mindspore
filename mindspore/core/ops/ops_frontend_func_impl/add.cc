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

#include <complex>
#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
template <typename T>
void ImplAdd(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<T *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = x1_data[i] + x2_data[i];
  }
}

using Handler = std::function<void(void *x1, void *x2, void *result, size_t size)>;
std::map<TypeId, Handler> add_impl_list = {{kNumberTypeBool, ImplAdd<bool>},
                                           {kNumberTypeInt8, ImplAdd<int8_t>},
                                           {kNumberTypeInt16, ImplAdd<int16_t>},
                                           {kNumberTypeInt32, ImplAdd<int32_t>},
                                           {kNumberTypeInt, ImplAdd<int>},
                                           {kNumberTypeInt64, ImplAdd<int64_t>},
                                           {kNumberTypeUInt, ImplAdd<u_int>},
                                           {kNumberTypeUInt8, ImplAdd<uint8_t>},
                                           {kNumberTypeUInt16, ImplAdd<uint16_t>},
                                           {kNumberTypeUInt32, ImplAdd<uint32_t>},
                                           {kNumberTypeUInt64, ImplAdd<uint64_t>},
                                           {kNumberTypeFloat16, ImplAdd<float16>},
                                           {kNumberTypeFloat32, ImplAdd<float>},
                                           {kNumberTypeFloat, ImplAdd<float>},
                                           {kNumberTypeFloat64, ImplAdd<double>},
                                           {kNumberTypeComplex64, ImplAdd<std::complex<float>>},
                                           {kNumberTypeComplex128, ImplAdd<std::complex<double>>}};

class AddFrontendFuncImpl : public OpFrontendFuncImpl {
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

    auto data_size = x1_tensor->DataSize();
    auto dtype = x1_tensor->data_type();
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, x1_shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    add_impl_list[dtype](x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Add", AddFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
