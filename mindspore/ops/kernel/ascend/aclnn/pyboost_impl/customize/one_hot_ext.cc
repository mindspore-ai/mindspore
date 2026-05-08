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

#include "kernel/ascend/aclnn/pyboost_impl/customize/one_hot_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/max.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr OneHotExtAscendCustomize(const TensorPtr &tensor_tensor, const Int64ImmPtr &num_classes,
                                           const TensorPtr &on_value, const TensorPtr &off_value,
                                           const Int64ImmPtr &axis) {
  static const int64_t MIN_DEPTH = 1;
  static const int64_t AUTO_DEPTH = -1;

  int64_t num_class_imm = GetValue<int64_t>(num_classes);
  if (num_class_imm == AUTO_DEPTH) {
    auto max_op = CREATE_PYBOOST_OP(Max, device::DeviceType::kAscend);
    auto max_tensor = max_op->Call(tensor_tensor);
    auto max_tensor_cpu = max_tensor->cpu();
    auto max_data = *static_cast<const int64_t *>(max_tensor_cpu->data_c());
    num_class_imm = max_data + 1;
    if (num_class_imm < MIN_DEPTH) {
      num_class_imm = MIN_DEPTH;
    }
  }
  auto num_class_new = std::make_shared<Int64Imm>(static_cast<int64_t>(num_class_imm));
  auto output = InferOutput<1>(prim::kPrimOneHotExt, tensor_tensor, num_class_new, on_value, off_value, axis);
  // No need to convert input
  TensorToDevice(tensor_tensor, on_value, off_value);
  TensorMalloc(output);
  constexpr int64_t axis_imm = -1;
  DISPATCH_ACLNN(aclnnOneHot, tensor_tensor, num_class_imm, on_value, off_value, axis_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
