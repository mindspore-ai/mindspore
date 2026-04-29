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

#include "kernel/ascend/aclnn/pyboost_impl/customize/uniform_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
double GetScalarValue(const std::shared_ptr<Scalar> &scalar) {
  if (scalar->isa<BoolImm>()) {
    return GetValue<bool>(scalar);
  } else if (scalar->isa<Int32Imm>()) {
    return GetValue<int32_t>(scalar);
  } else if (scalar->isa<Int64Imm>()) {
    return GetValue<int64_t>(scalar);
  } else if (scalar->isa<FP32Imm>()) {
    return GetValue<float>(scalar);
  } else if (scalar->isa<FP64Imm>()) {
    return GetValue<double>(scalar);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported type: " << scalar->type_name();
  }
}
}  // namespace

tensor::TensorPtr UniformExtAscendCustomize(const TensorPtr &tensor_tensor, const ScalarPtr &a, const ScalarPtr &b,
                                            const TensorPtr &seed, const TensorPtr &offset) {
  MS_LOG(DEBUG) << "UniformExt call start";
  auto output = InferOutput<1>(prim::kPrimUniformExt, tensor_tensor, a, b, seed, offset);

  double a_imm = GetScalarValue(a);
  double b_imm = GetScalarValue(b);

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  TensorToDevice(tensor_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnInplaceUniform, output, a_imm, b_imm, static_cast<uint64_t>(seed_imm),
                 static_cast<uint64_t>(offset_imm));
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
