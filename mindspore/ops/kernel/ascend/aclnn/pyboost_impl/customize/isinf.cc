/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/isinf.h"

#include <limits>

#include "include/pynative/utils/pyboost/auto_generate/abs.h"
#include "include/pynative/utils/pyboost/auto_generate/equal.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IsInfAscendCustomize(const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "IsInfCustomize start";
  auto output = InferOutput<1>(prim::kPrimIsInf, input_tensor);

  auto input_type = input_tensor->Dtype()->type_id();
  bool is_int_type = (input_type >= kNumberTypeBool) && (input_type < kNumberTypeFloat);

  TensorPtr abs_out = input_tensor;
  if (!is_int_type) {
    const auto abs_op = CREATE_PYBOOST_OP(Abs, device::DeviceType::kAscend);
    abs_out = abs_op->Call(input_tensor);
    TensorToDevice(abs_out);
  }

  TensorMalloc(output);

  if (is_int_type) {
    DISPATCH_ACLNN(aclnnInplaceZero, output);
    return output;
  }

  switch (input_type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32: {
      static const ScalarPtr inf_value = std::make_shared<FP32Imm>(std::numeric_limits<float>::infinity());
      DISPATCH_ACLNN(aclnnEqScalar, abs_out, inf_value, output);
      break;
    }
    case kNumberTypeFloat64: {
      static const ScalarPtr inf_value = std::make_shared<FP64Imm>(std::numeric_limits<double>::infinity());
      DISPATCH_ACLNN(aclnnEqScalar, abs_out, inf_value, output);
      break;
    }
    case kNumberTypeBFloat16: {
      static const ScalarPtr inf_value = std::make_shared<BF16Imm>(std::numeric_limits<BFloat16>::infinity());
      DISPATCH_ACLNN(aclnnEqScalar, abs_out, inf_value, output);
      break;
    }
    case kNumberTypeFloat16: {
      static const ScalarPtr inf_value =
        std::make_shared<FP32Imm>(static_cast<float>(std::numeric_limits<float>::infinity()));
      DISPATCH_ACLNN(aclnnEqScalar, abs_out, inf_value, output);
      break;
    }
    default:
      MS_EXCEPTION(TypeError) << "Op IsInf does not support type " << TypeIdToString(input_type);
      break;
  }
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
