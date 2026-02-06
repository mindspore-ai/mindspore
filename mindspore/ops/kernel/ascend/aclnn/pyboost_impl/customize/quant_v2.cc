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

#include "kernel/ascend/aclnn/pyboost_impl/customize/quant_v2.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
using mindspore::device::ascend::AscendQuantRoundMode;
namespace kernel {
namespace pyboost {
tensor::TensorPtr QuantV2AscendCustomize(const tensor::TensorPtr &x_tensor, const tensor::TensorPtr &scale_tensor,
                                         const tensor::TensorPtr &offset_tensor, const BoolImmPtr &sqrt_mode,
                                         const Int64ImmPtr &rounding_mode, const Int64ImmPtr &dst_type) {
  auto output =
    InferOutput<1>(prim::kPrimQuantV2, x_tensor, scale_tensor, offset_tensor, sqrt_mode, rounding_mode, dst_type);

  auto sqrt_mode_imm = GetValue<bool>(sqrt_mode);
  auto rounding_mode_imm = GetValue<int64_t>(rounding_mode);
  std::string rounding_mode_str = AscendQuantRoundMode::ConvertEnumToString(rounding_mode_imm);

  TypeId out_type = output->data_type();

  TensorToDevice(x_tensor, scale_tensor, offset_tensor);
  TensorAllocate(output);

  DISPATCH_ACLNN(aclnnAscendQuant, x_tensor, scale_tensor, offset_tensor, sqrt_mode_imm, rounding_mode_str, out_type,
                 output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
