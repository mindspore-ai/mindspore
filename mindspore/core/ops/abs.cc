/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/abs.h"

#include <memory>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class AbsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
    auto shape_element = input_args[0]->BuildShape();
    MS_EXCEPTION_IF_NULL(shape_element);
    return shape_element;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
    auto x_type = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim->name());
    return x_type;
  }

  ValuePtr InferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      return nullptr;
    }

    auto x = input_args[0]->BuildValue();
    if (x == nullptr) {
      return nullptr;
    }
    auto x_tensor = x->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }

    auto data_size = x_tensor->DataSize();
    auto dtype = x_tensor->data_type();
    auto shape = InferShape(prim, input_args)->cast<abstract::ShapePtr>();
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape->shape());
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
        MS_EXCEPTION(TypeError) << "For '" << prim->name()
                                << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
                                   "'uint16','uint32', 'uint64','float16', 'float32', 'float64'], but got: "
                                << x_tensor->ToString() << ".";
      }
    }
    return result_tensor;
  }

 private:
  template <typename T>
  void ImpleAbs(void *origin, void *target, size_t size) const {
    MS_EXCEPTION_IF_NULL(origin);
    MS_EXCEPTION_IF_NULL(target);
    auto origin_data = reinterpret_cast<T *>(origin);
    auto target_data = reinterpret_cast<T *>(target);
    auto zero_val = static_cast<T>(0);
    for (size_t i = 0; i < size; ++i) {
      target_data[i] = origin_data[i] >= zero_val ? origin_data[i] : -origin_data[i];
    }
  }
};

MIND_API_OPERATOR_IMPL(Abs, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Abs, prim::kPrimAbs, AbsInfer, true);
}  // namespace ops
}  // namespace mindspore
