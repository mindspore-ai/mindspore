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

#include "kernel/ascend/aclnn/pyboost_impl/customize/elu_grad_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr EluGradExtAscendCustomize(const TensorPtr &dy_tensor, const TensorPtr &x_or_out_tensor,
                                            const ScalarPtr &alpha, const BoolImmPtr &is_result) {
  auto output = InferOutput<1>(prim::kPrimEluGradExt, dy_tensor, x_or_out_tensor, alpha, is_result);
  const auto is_result_imm = GetValue<bool>(is_result);

  TensorToDevice(dy_tensor, x_or_out_tensor);
  TensorMalloc(output);

  // Async
  static const ScalarPtr scale = std::make_shared<FP32Imm>(1.f);
  static const ScalarPtr input_scale = std::make_shared<FP32Imm>(1.f);

  DISPATCH_ACLNN(aclnnEluBackward, dy_tensor, alpha, scale, input_scale, is_result_imm, x_or_out_tensor, output);
  MS_LOG(DEBUG) << "Run device task EluGradExt end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
