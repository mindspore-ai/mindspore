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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/conv_pool_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API PoolingInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: a tensor.
    auto input_tensor = input_args[kIndex0];
    auto input_shape = dyn_cast<abstract::TensorShape>(input_tensor->GetShape());  // NCHW
    MS_EXCEPTION_IF_NULL(input_shape);
    const size_t input_shape_size = 4;
    if (input_shape->shape().size() != input_shape_size) {
      MS_LOG(EXCEPTION) << "Pooling input should be a 4-D tensor.";
    }
    const size_t H_INDEX = 2;
    const size_t W_INDEX = 3;
    int64_t h_input = input_shape->shape()[H_INDEX];
    int64_t w_input = input_shape->shape()[W_INDEX];

    int64_t window = GetValue<int64_t>(primitive->GetAttr("window"));
    int64_t stride = GetValue<int64_t>(primitive->GetAttr("stride"));
    int64_t padding = GetValue<int64_t>(primitive->GetAttr("pad"));
    int64_t nan_opt = GetValue<int64_t>(primitive->GetAttr("nan_opt"));
    int64_t data_mode = GetValue<int64_t>(primitive->GetAttr("data_mode"));
    int64_t ceil_mode = GetValue<int64_t>(primitive->GetAttr("ceil_mode"));

    if (stride <= 0) {
      MS_LOG(EXCEPTION) << "Invalid stride value: " << stride << ", should greater then 0";
    }
    if (nan_opt != 0) {
      MS_LOG(EXCEPTION) << "Invalid nan_opt value: " << nan_opt << ", should be 0";
    }
    if (data_mode != 1) {
      MS_LOG(EXCEPTION) << "Invalid data_mode value: " << data_mode << ", should be 1";
    }
    if (ceil_mode != 0) {
      MS_LOG(EXCEPTION) << "Invalid ceil_mode value: " << ceil_mode << ", should be 0";
    }

    auto pad_mode_ptr = primitive->GetAttr("pad_mode");
    if (pad_mode_ptr != nullptr) {
      int64_t pad_mode;
      const int64_t middle = 2;
      CheckAndConvertUtils::GetPadModEnumValue(pad_mode_ptr, &pad_mode, true);
      if (pad_mode == static_cast<int64_t>(PadMode::VALID)) {
        padding = 0;
      } else if (pad_mode == static_cast<int64_t>(PadMode::SAME)) {
        padding = (window - 1) / middle;
      }
    }
    std::set<std::string> available_mode{"max", "avg"};
    auto mode_ptr = primitive->GetAttr("mode");
    if ((mode_ptr != nullptr) && mode_ptr->isa<StringImm>()) {
      auto mode = mode_ptr->cast<StringImmPtr>()->value();
      if (available_mode.find(mode) == available_mode.end()) {
        MS_LOG(EXCEPTION) << "Unsupported pooling mode: " << mode << ".";
      }
    }
    const int64_t twice = 2;
    int64_t h_out = (((h_input + twice * padding - (window - 1)) - 1) / stride) + 1;
    int64_t w_out = (((w_input + twice * padding - (window - 1)) - 1) / stride) + 1;
    ShapeVector shape_out = {input_shape->shape()[0], input_shape->shape()[1], h_out, w_out};
    return std::make_shared<abstract::TensorShape>(shape_out);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    auto input_tensor = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    (void)CheckTensorDType(input_tensor, {kFloat16, kFloat32}, "Input 0 of Pooling should be %s");
    return input_tensor->GetType()->Clone();
  }
};

class MIND_API Pooling : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Pooling);
  /// \brief Constructor.
  Pooling() : BaseOperator("Pooling") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Pooling, prim::kPrimPooling, PoolingInfer, false);
}  // namespace ops
}  // namespace mindspore
