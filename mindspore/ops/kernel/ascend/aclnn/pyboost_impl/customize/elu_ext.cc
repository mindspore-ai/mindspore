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

#include "kernel/ascend/aclnn/pyboost_impl/customize/elu_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr EluExtAscendCustomize(const TensorPtr &input, const ScalarPtr &alpha) {
  MS_LOG(DEBUG) << "Call EluExt start";
  TypeId data_type = input->data_type();
  if (data_type == kNumberTypeFloat64) {
    MS_LOG(EXCEPTION) << "Unsupported input dtype: float64, because aclnnEluBackward does not support dtype: float64";
  }
  auto output = InferOutput<1>(prim::kPrimEluExt, input, alpha);

  TensorToDevice(input);
  TensorMalloc(output);

  // Async
  static const ScalarPtr scale = std::make_shared<FP32Imm>(1.f);
  static const ScalarPtr input_scale = std::make_shared<FP32Imm>(1.f);
  DISPATCH_ACLNN(aclnnElu, input, alpha, scale, input_scale, output);
  MS_LOG(DEBUG) << "Call EluExt end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
