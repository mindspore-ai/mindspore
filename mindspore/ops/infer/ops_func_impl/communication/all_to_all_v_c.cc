/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/communication/all_to_all_v_c.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/helper.h"
#include "primitive/other_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> Tranpose1d(int64_t rank_size, const std::vector<int64_t> &list_1d) {
  int64_t size = list_1d.size();
  if (rank_size * rank_size != size) {
    MS_EXCEPTION(ValueError) << "The size of the one-dimensional array cannot form a square matrix.";
  }
  std::vector<int64_t> transposed(rank_size * rank_size);
  for (int64_t i = 0; i < rank_size; ++i) {
    for (int64_t j = 0; j < rank_size; ++j) {
      transposed[j * rank_size + i] = list_1d[i * rank_size + j];
    }
  }
  return transposed;
}

int64_t GetOutputNumel(int64_t block_size, int64_t rank_id, int64_t rank_size, const std::vector<int64_t> &list_1d) {
  int64_t size = list_1d.size();
  if (rank_size * rank_size != size) {
    MS_EXCEPTION(ValueError) << "The size of the one-dimensional array cannot form a square matrix.";
  }
  int64_t output_numel = 0;
  for (int64_t i = 0; i < rank_size; ++i) {
    for (int64_t j = 0; j < rank_size; ++j) {
      if (j == rank_id) {
        output_numel += list_1d[i * rank_size + j] * block_size;
      }
    }
  }

  return output_numel;
}

}  // namespace
MIND_API_OPERATOR_IMPL(AlltoAllVC, BaseOperator);
class AlltoAllVCInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    BaseShapePtr shape;
    int64_t output_numel = 0;
    auto rank_size_ptr = primitive->GetAttr(kRankSize);
    auto rank_size = GetValue<int64_t>(rank_size_ptr);
    auto rank_id_ptr = primitive->GetAttr(kRankId);
    auto rank_id = GetValue<int64_t>(rank_id_ptr);
    MS_LOG(DEBUG) << "For '" << prim_name << "', input rank_id : " << rank_id << ".";

    if (input_args.size() == kInputNum2) {
      (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                               prim_name);
      auto value = GetArrayValue<int64_t>(input_args[kIndex1]);
      if (!value.has_value()) {
        return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
      }
      if (value.value().HasUnknownValue()) {
        MS_EXCEPTION(ValueError)
          << "For primitive[" << prim_name
          << "], there are unknown values in input1, please handle this case before calling this function.";
      }
      auto values_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
      if (values_shape.size() != 1) {
        MS_EXCEPTION(ValueError) << "For AlltoAllVC, input0 shape must be a 1-D tensor"
                                 << ", but got shape size " << values_shape.size() << ".";
      }
      auto block_size = GetValue<int64_t>(primitive->GetAttr(kAttrBlockSize));
      auto transpose_ptr = primitive->GetAttr(kAttrTransPose);
      std::vector<int64_t> list_1d = value.value().ToVector();
      auto transpose = GetValue<bool>(transpose_ptr);
      MS_LOG(DEBUG) << "For '" << prim_name << "', input transpose : " << transpose << ".";
      std::vector<int64_t> transposed;
      if (transpose) {
        transposed = Tranpose1d(rank_size, list_1d);
      } else {
        transposed = list_1d;
      }
      output_numel = GetOutputNumel(block_size, rank_id, rank_size, transposed);
    } else {
      MS_LOG(EXCEPTION) << "AlltoAllVC input numbers must be 2.";
    }
    if (output_numel == 0) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{});
    }
    return std::make_shared<abstract::TensorShape>(ShapeVector{output_numel});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    const auto prim_name = prim->name();
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->GetType();

    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input0 must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }
    // flag to check different valid types on ascend
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

    if (!is_ascend) {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types_with_bool, prim_name);
    } else {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types, prim_name);
    }
    MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
    auto list_type = input_args[kIndex1]->GetType();
    MS_EXCEPTION_IF_NULL(list_type);
    if (list_type->isa<TensorType>()) {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("send_count_matrix", list_type, {kInt64}, prim_name);
    } else if (list_type->isa<Tuple>() || list_type->isa<List>()) {
      (void)CheckAndConvertUtils::CheckIntOrTupleInt("send_count_matrix", input_args[1], prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "input1 must be a int in Tensor or list.";
    }
    return x_type->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    if (input_args.size() == kInputNum2) {
      (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                               prim_name);
    } else {
      MS_LOG(EXCEPTION) << "AlltoAllVC input numbers must be 2.";
    }
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(AlltoAllVC, prim::kPrimAlltoAllVC, AlltoAllVCInfer, false);
}  // namespace ops
}  // namespace mindspore
