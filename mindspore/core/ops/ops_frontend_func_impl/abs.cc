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

#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
template <typename T>
void ImpleAbs(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  auto zero_val = static_cast<T>(0);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = origin_data[i] >= zero_val ? origin_data[i] : -origin_data[i];
  }
}

class AbsFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x = input_args[kIndex0]->GetValue();
    if (x == nullptr || x->ContainsValueAny()) {
      return nullptr;
    }
    auto x_tensor = x->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }

    auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x_shape)) {
      return nullptr;
    }

    auto data_size = x_tensor->DataSize();
    auto dtype = x_tensor->data_type();
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, x_shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto x_datac = x_tensor->data_c();
    auto result_datac = result_tensor->data_c();
    switch (dtype) {
      case kNumberTypeInt8: {
        ImpleAbs<int8_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt16: {
        ImpleAbs<int16_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt32: {
        ImpleAbs<int32_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt64: {
        ImpleAbs<int64_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeUInt8: {
        ImpleAbs<uint8_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeUInt16: {
        ImpleAbs<uint16_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeUInt32: {
        ImpleAbs<uint32_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeUInt64: {
        ImpleAbs<uint64_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat16: {
        ImpleAbs<float16>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat32: {
        ImpleAbs<float>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat64: {
        ImpleAbs<double>(x_datac, result_datac, data_size);
        break;
      }
      default: {
        MS_LOG(DEBUG) << "For 'Abs', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
                         "'uint16','uint32', 'uint64','float16', 'float32', 'float64'], but got: "
                      << x_tensor->ToString() << ".";
        return nullptr;
      }
    }
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Abs", AbsFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
