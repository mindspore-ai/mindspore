/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/cudnn_gru.h"

#include <unordered_map>
#include <map>
#include <string>
#include <set>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kCudnnGRUInputDim = 3;
constexpr size_t kCudnnGRUHDim = 3;
constexpr int64_t kCudnnGRUInputsNum = 3;
constexpr auto kCudnnGRURealNumLayers = "real_num_layers";
constexpr auto kCudnnGRURealHiddenSize = "real_hidden_size";

std::unordered_map<std::string, int64_t> CudnnGRUGetAttrMap(const PrimitivePtr &primitive) {
  std::unordered_map<std::string, int64_t> attr_map;
  auto input_size_ptr = primitive->GetAttr(kInputSize);
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  attr_map[kInputSize] = GetValue<int64_t>(input_size_ptr);

  auto hidden_size_ptr = primitive->GetAttr(kHiddenSize);
  MS_EXCEPTION_IF_NULL(hidden_size_ptr);
  auto hidden_size = GetValue<int64_t>(hidden_size_ptr);
  attr_map[kHiddenSize] = hidden_size;

  auto num_layers_ptr = primitive->GetAttr(kNumLayers);
  MS_EXCEPTION_IF_NULL(num_layers_ptr);
  auto num_layers = GetValue<int64_t>(num_layers_ptr);

  auto bidirectional_ptr = primitive->GetAttr(kBidirectional);
  MS_EXCEPTION_IF_NULL(bidirectional_ptr);
  auto bidirectional = GetValue<bool>(bidirectional_ptr);

  auto real_hidden_size = bidirectional ? hidden_size * 2 : hidden_size;
  auto real_num_layers = bidirectional ? num_layers * 2 : num_layers;
  attr_map[kCudnnGRURealNumLayers] = real_num_layers;
  attr_map[kCudnnGRURealHiddenSize] = real_hidden_size;
  return attr_map;
}

abstract::TupleShapePtr CudnnGRUInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kCudnnGRUInputsNum, op_name);
  auto attr_map = CudnnGRUGetAttrMap(primitive);
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto h_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto input_shape = input_shape_map[kShape];  // (seq_len, batch_size, input_size)
  auto h_shape = h_shape_map[kShape];          // (real_num_layers, batch_size, hidden_size)

  int64_t seq_len = abstract::Shape::kShapeDimAny;
  int64_t batch_size = abstract::Shape::kShapeDimAny;
  if (!IsDynamicRank(input_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("input_dims", SizeToLong(input_shape.size()), kEqual, kCudnnGRUInputDim,
                                             op_name);
    seq_len = input_shape[kInputIndex0];
    batch_size = input_shape[kInputIndex1];
    if (input_shape[kInputIndex2] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("input_shape[2]", input_shape[kInputIndex2], kEqual,
                                               attr_map[kInputSize]);
    }
  }

  if (!IsDynamicRank(h_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("h_dims", SizeToLong(h_shape.size()), kEqual, kCudnnGRUHDim, op_name);
    if (h_shape[kInputIndex0] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("h_shape[0]", h_shape[kInputIndex0], kEqual,
                                               attr_map[kCudnnGRURealNumLayers], op_name);
    }
    if (h_shape[kInputIndex1] != abstract::Shape::kShapeDimAny) {
      if (batch_size != abstract::Shape::kShapeDimAny && batch_size != h_shape[kInputIndex1]) {
        MS_LOG(EXCEPTION) << "For " << op_name << ", input_shape[1] and h_shape[1] should be -1 or equal, but got "
                          << batch_size << " and " << h_shape[kInputIndex1] << ".";
      }
      batch_size = h_shape[kInputIndex1];
    }
    if (h_shape[kInputIndex2] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("h_shape[2]", h_shape[kInputIndex2], kEqual, attr_map[kHiddenSize],
                                               op_name);
    }
  }

  auto output_shape_ptr =
    std::make_shared<abstract::Shape>(ShapeVector{seq_len, batch_size, attr_map[kCudnnGRURealHiddenSize]});
  auto hn_shape_ptr =
    std::make_shared<abstract::Shape>(ShapeVector{attr_map[kCudnnGRURealNumLayers], batch_size, attr_map[kHiddenSize]});
  auto reserve_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{1, 1});
  auto state_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{1, 1});

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output_shape_ptr, hn_shape_ptr, reserve_shape_ptr, state_shape_ptr});
}

TuplePtr CudnnGRUInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kFloat32};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("h", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("w", input_args[kInputIndex2]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(CudnnGRU, BaseOperator);
class MIND_API CudnnGRUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CudnnGRUInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kCudnnGRUInputsNum, primitive->name());
    return CudnnGRUInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CudnnGRU, prim::kPrimCudnnGRU, CudnnGRUInfer, false);
}  // namespace ops
}  // namespace mindspore
