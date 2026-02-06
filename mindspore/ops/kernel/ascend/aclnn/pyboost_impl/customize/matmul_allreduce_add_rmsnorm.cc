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
#include "kernel/ascend/aclnn/pyboost_impl/customize/matmul_allreduce_add_rmsnorm.h"
#include <string>
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> MatmulAllReduceAddRmsNormAscendCustomize(
  const TensorPtr &x1_tensor, const TensorPtr &x2_tensor, const TensorPtr &bias_tensor,
  const TensorPtr &residual_tensor, const TensorPtr &gamma_tensor, const FP32ImmPtr &epsilon, const StringImmPtr &group,
  const Int64ImmPtr &reduction, const Int64ImmPtr &comm_turn, const Int64ImmPtr &stream_mode) {
  MS_LOG(DEBUG) << "MatmulAllReduceAddRmsNormAscendCustomize call start.";
  auto outputs = InferOutput<2>(prim::kPrimMatmulAllReduceAddRmsNorm, x1_tensor, x2_tensor, bias_tensor,
                                residual_tensor, gamma_tensor, epsilon, group, reduction, comm_turn, stream_mode);

  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));
  auto group_str = GetValue<std::string>(group);
  std::string group_imm = device::ascend::OpApiUtil::GetCommName(group_str);
  auto comm_turn_imm = GetValue<int64_t>(comm_turn);
  auto stream_mode_imm = GetValue<int64_t>(stream_mode);

  // transform reduction enum value to corresponding value
  auto reduction_value = device::ascend::GEReduction::ConvertEnumToString(GetValue<int64_t>(reduction));

  TensorToDevice(x1_tensor, x2_tensor, bias_tensor, residual_tensor, gamma_tensor);
  TensorAllocate(outputs);

  auto [o0, o1] = outputs;

  // for empty bias input
  if (bias_tensor->ElementsNum() == 0) {
    DISPATCH_ACLNN(aclnnMatmulAllReduceAddRmsNorm, x1_tensor, x2_tensor, nullptr, residual_tensor, gamma_tensor,
                   epsilon_imm, group_imm, reduction_value, comm_turn_imm, stream_mode_imm, o0, o1);
  } else {
    DISPATCH_ACLNN(aclnnMatmulAllReduceAddRmsNorm, x1_tensor, x2_tensor, bias_tensor, residual_tensor, gamma_tensor,
                   epsilon_imm, group_imm, reduction_value, comm_turn_imm, stream_mode_imm, o0, o1);
  }
  MS_LOG(DEBUG) << "MatmulAllReduceAddRmsNormAscendCustomize call end.";
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
