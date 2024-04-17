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

#include <algorithm>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
template <typename T>
void ImpleNeg(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = -origin_data[i];
  }
}

using NegHandler = std::function<void(void *origin, void *target, size_t size)>;
std::map<TypeId, NegHandler> neg_impl_list = {{kNumberTypeInt8, ImpleNeg<int8_t>},
                                              {kNumberTypeInt16, ImpleNeg<int16_t>},
                                              {kNumberTypeInt32, ImpleNeg<int32_t>},
                                              {kNumberTypeInt64, ImpleNeg<int64_t>},
                                              {kNumberTypeUInt8, ImpleNeg<uint8_t>},
                                              {kNumberTypeUInt16, ImpleNeg<uint16_t>},
                                              {kNumberTypeUInt32, ImpleNeg<uint32_t>},
                                              {kNumberTypeUInt64, ImpleNeg<uint64_t>},
                                              {kNumberTypeFloat16, ImpleNeg<float16>},
                                              {kNumberTypeFloat32, ImpleNeg<float>},
                                              {kNumberTypeFloat64, ImpleNeg<double>},
                                              {kNumberTypeComplex64, ImpleNeg<std::complex<float>>},
                                              {kNumberTypeComplex128, ImpleNeg<std::complex<double>>}};

class NegFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x = input_args[kInputIndex0]->GetValue();
    if (x == nullptr) {
      return nullptr;
    }
    auto x_tensor = x->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }

    auto data_size = x_tensor->DataSize();
    auto dtype = x_tensor->data_type();
    auto shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(shape)) {
      return nullptr;
    }
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
    auto iter = neg_impl_list.find(dtype);
    if (iter == neg_impl_list.end()) {
      MS_LOG(DEBUG)
        << "For '" << primitive->name()
        << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
           "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
        << x_tensor->ToString() << ".";
      return nullptr;
    }
    auto x_datac = x_tensor->data_c();
    auto result_datac = result_tensor->data_c();
    iter->second(x_datac, result_datac, data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Neg", NegFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
