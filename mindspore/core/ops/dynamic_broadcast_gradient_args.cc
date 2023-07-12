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

#include "ops/dynamic_broadcast_gradient_args.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
const int kInputNum = 2;
const size_t one = 1;

int64_t CheckInputsAndGetShape(const AbstractBasePtr &input_arg, const string &prim_name) {
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<abstract::AbstractTensor>()) {
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_arg->BuildShape())[kShape];
    auto input_size = input_shape.size();
    if (input_size != 1) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input shape must be 1-D, but got: " << input_size << "-D.";
    }
    return input_shape[0];
  } else if (input_arg->isa<abstract::AbstractTuple>()) {
    auto x_shape = dyn_cast<abstract::AbstractTuple>(input_arg);
    auto x_shape_data = x_shape->elements();
    return SizeToLong(x_shape_data.size());
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input type must be a tuple or Tensor.";
  }
}

abstract::TupleShapePtr Infer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto x_shape0 = CheckInputsAndGetShape(input_args[0], prim_name);
  auto y_shape0 = CheckInputsAndGetShape(input_args[1], prim_name);
  ShapeVector shape{abstract::Shape::kShapeDimAny};
  ShapeVector max_shape;
  // DynamicBroadcastGradientArgs is a compute depend op
  if (x_shape0 >= 0 && y_shape0 >= 0) {
    max_shape = {x_shape0 > y_shape0 ? x_shape0 : y_shape0};
    // Currently, if the max_shape is 0, there may be some problems
    max_shape[0] = max_shape[0] != 0 ? max_shape[0] : 1;
  }

  auto out_shape = std::make_shared<abstract::Shape>(shape, max_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}

void UpdatePreIsOne(std::vector<bool> *const prev_is_one, std::vector<bool> current_is_one) {
  for (size_t i = 0; i < kInputNum; ++i) {
    (*prev_is_one)[i] = current_is_one[i];
  }
}

void AddElementToGradReduceIdx(std::vector<std::vector<int64_t>> *const grad_reduce_idx,
                               std::vector<bool> current_is_one, bool none_is_one, const size_t largest_rank,
                               size_t j) {
  MS_EXCEPTION_IF_NULL(grad_reduce_idx);
  for (size_t i = 0; i < kInputNum; ++i) {
    if (current_is_one[i] && !none_is_one) {
      (void)(*grad_reduce_idx)[i].emplace_back(SizeToLong(largest_rank - one - j));
    }
  }
}

std::vector<std::vector<int64_t>> GetGradientIndices(const std::vector<std::vector<int64_t>> &reverse_shape,
                                                     const size_t largest_rank) {
  std::vector<std::vector<int64_t>> grad_reduce_idx(kInputNum);
  // indices of j-th component of each input.
  std::vector<bool> prev_is_one(kInputNum);
  std::vector<bool> current_is_one(kInputNum);
  for (size_t i = 0; i < kInputNum; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }

  bool set_one = false;
  for (size_t j = 0; j < largest_rank; ++j) {
    int output_dim = -1;
    bool output_dim_set = false;
    bool none_is_one = true;
    // Find which indices are 1.
    for (size_t i = 0; i < kInputNum; ++i) {
      if (reverse_shape[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || reverse_shape[i][j] == static_cast<int64_t>(output_dim)) {
          output_dim = LongToInt(reverse_shape[i][j]);
          output_dim_set = true;
        } else {
          MS_LOG(EXCEPTION) << "Input[0] and input[1] Cannot broadcast!";
        }
      }
    }
    // All dimensions are 1.
    if (!output_dim_set) {
      for (size_t i = 0; i < kInputNum; ++i) {
        (void)grad_reduce_idx[i].emplace_back(SizeToLong(largest_rank - one - j));
      }
      continue;
    } else if (std::equal(current_is_one.begin(), current_is_one.end(), prev_is_one.begin()) && set_one) {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    } else {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    }
    set_one = true;
    UpdatePreIsOne(&prev_is_one, current_is_one);
  }
  return grad_reduce_idx;
}

std::vector<std::vector<int64_t>> CalculateOutput(const std::vector<std::vector<int64_t>> &x) {
  std::vector<std::vector<int64_t>> grad_reduce_idx(kInputNum);
  bool all_equal = true;
  size_t largest_rank = 0;
  for (size_t i = 0; i < kInputNum; ++i) {
    if (x[i] != x[0]) {
      all_equal = false;
    }
    if (x[i].size() > largest_rank) {
      largest_rank = x[i].size();
    }
  }
  if (all_equal) {
    return grad_reduce_idx;
  }

  // Reverse input the shapes
  std::vector<std::vector<int64_t>> reverse_shape(kInputNum);
  for (size_t i = 0; i < kInputNum; ++i) {
    reverse_shape[i] = x[i];
    std::reverse(reverse_shape[i].begin(), reverse_shape[i].end());
  }

  // 1-extend and align all vectors.
  for (size_t i = 0; i < kInputNum; ++i) {
    if (reverse_shape[i].size() < largest_rank) {
      reverse_shape[i].resize(largest_rank, 1);
    }
  }
  grad_reduce_idx = GetGradientIndices(reverse_shape, largest_rank);
  return grad_reduce_idx;
}
}  // namespace

class MIND_API DynamicBroadcastGradientArgsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Infer(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) const override {
    auto types = std::vector<TypePtr>{kInt64, kInt64};
    auto output_type = std::make_shared<Tuple>(types);
    return output_type;
  }

  ValuePtr InferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    if (input_args.empty()) {
      MS_LOG(ERROR) << "DynamicBroadcastGradientArgs input args dose not exist.";
      return nullptr;
    }

    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    std::vector<std::vector<int64_t>> input_shapes(kInputNum);
    auto x = input_args[0]->BuildValue();
    if (x == kValueAny) {
      MS_LOG(INFO) << "DynamicBroadcastGradientArgs input_0 is ValueAny, will backoff to cpu.";
      return nullptr;
    }
    input_shapes[0] = GetValue<std::vector<int64_t>>(x);

    auto y = input_args[1]->BuildValue();
    if (y == kValueAny) {
      MS_LOG(INFO) << "DynamicBroadcastGradientArgs input_1 is ValueAny, will backoff to cpu.";
      return nullptr;
    }
    input_shapes[1] = GetValue<std::vector<int64_t>>(y);
    auto grad_reduce_idx = CalculateOutput(input_shapes);
    ValuePtr res = MakeValue(grad_reduce_idx);
    return res;
  }
};

MIND_API_OPERATOR_IMPL(DynamicBroadcastGradientArgs, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicBroadcastGradientArgs, prim::kPrimDynamicBroadcastGradientArgs,
                                 DynamicBroadcastGradientArgsInfer, true);
}  // namespace ops
}  // namespace mindspore
