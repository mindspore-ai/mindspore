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

#include "infer/ops_func_impl/communication/op_comm_func_impl.h"
#include <string>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

uint64_t CheckRankSize(const std::string &name, const std::unique_ptr<InferInfo> &value) {
  auto rank_size_opt = value->GetScalarValue<int64_t>();
  MS_CHECK_VALUE(rank_size_opt.has_value(), name + " error: rank_size input should has valid value.");
  auto rank_size = rank_size_opt.value();
  if (rank_size <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << name << "', input rank_size must > 0, but got: " << rank_size << ".";
  }
  return rank_size;
}

uint64_t GetRankValue(const std::string &name, const std::unique_ptr<InferInfo> &value) {
  auto rank = value->GetScalarValue<int64_t>();
  MS_CHECK_VALUE(rank.has_value(), name + " error: rank input should has valid value.");
  return rank.value();
}

void CheckInferShape(const std::string &name, const ShapeVector &input_shape, const ShapeVector &output_shape) {
  if (input_shape.size() != output_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << name << "', output_tensor shape size must be equal to " << input_shape.size()
                             << ", but got " << output_shape.size();
  }
  if (!std::equal(output_shape.begin(), output_shape.end(), input_shape.begin())) {
    MS_EXCEPTION(ValueError) << "For '" << name << "', output_tensor shape must be equal to " << input_shape
                             << ", but got " << output_shape;
  }
}

TypeId CheckInferType(const std::string &name, const TypeId type) {
  static const std::set<TypeId> valid_types = {
    kNumberTypeBool,    kNumberTypeInt8,     kNumberTypeUInt8,  kNumberTypeInt16,  kNumberTypeUInt16,
    kNumberTypeInt32,   kNumberTypeUInt32,   kNumberTypeInt64,  kNumberTypeUInt64, kNumberTypeFloat16,
    kNumberTypeFloat32, kNumberTypeBFloat16, kNumberTypeFloat64};
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", type, valid_types, name);
  return type;
}

TypeId CheckReduceInferType(const std::string &name, const TypeId type) {
  static const std::set<TypeId> valid_types = {kNumberTypeBool,    kNumberTypeInt8,    kNumberTypeInt16,
                                               kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeFloat16,
                                               kNumberTypeFloat32, kNumberTypeBFloat16};
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", type, valid_types, name);
  return type;
}

TypeId CheckInferTypes(const std::string &name, const TypeId type, const TypeId out_type, bool is_reduce_op) {
  if (is_reduce_op) {
    CheckReduceInferType(name, type);
  } else {
    CheckInferType(name, type);
  }

  if (out_type != type) {
    MS_EXCEPTION(ValueError) << "For '" << name << "', output_tensor type " << TypeIdToString(out_type)
                             << " must be equal to input_tensor type " << TypeIdToString(type);
  }
  return type;
}

}  // namespace ops
}  // namespace mindspore
