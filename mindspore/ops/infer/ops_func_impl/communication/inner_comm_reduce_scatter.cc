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

#include "infer/ops_func_impl/communication/inner_comm_reduce_scatter.h"
#include <memory>
#include <set>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr InnerCommReduceScatterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto rank_size_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  MS_CHECK_VALUE(rank_size_opt.has_value(), primitive->name() + " error: rank_size input should has valid value.");
  auto rank_size = rank_size_opt.value();
  if (rank_size <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input rank_size must > 0, but got: " << rank_size
                             << ".";
  }

  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  MS_LOG(INFO) << "For '" << primitive->name() << "', input rank_size : " << rank_size
               << ", input shape:" << input_shape;

  if (input_shape[0] < 0) {  // first dim is dynamic shape
    return std::make_shared<abstract::TensorShape>(input_shape);
  }
  if (input_shape[0] % rank_size != 0) {
    MS_EXCEPTION(ValueError)
      << "the first dimension for 'input_shape' must be divided by 'rank_size', but got input_shape[0]: "
      << input_shape[0] << ", rank_size: " << rank_size;
  }
  input_shape[0] = static_cast<int64_t>(input_shape[0] / rank_size);
  return std::make_shared<abstract::TensorShape>(input_shape);
}

TypePtr InnerCommReduceScatterFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto type = input_args[kIndex0]->GetType();
  const std::set<TypePtr> default_valid_types = {kInt8, kInt32, kFloat16, kFloat32, kBFloat16, kComplex64};
  const std::set<TypePtr> gpu_valid_types = {kBool,   kInt8,    kInt32,   kUInt32, kInt64,
                                             kUInt64, kFloat16, kFloat32, kFloat64};
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, gpu_valid_types, primitive->name());
  } else {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, default_valid_types, primitive->name());
  }
  return type;
}
}  // namespace ops
}  // namespace mindspore
