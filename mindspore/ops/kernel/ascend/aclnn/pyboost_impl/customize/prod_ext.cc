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

#include "kernel/ascend/aclnn/pyboost_impl/customize/prod_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ProdExtAscendCustomize(const TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &axis,
                                         const BoolImmPtr &keep_dims, const std::optional<Int64ImmPtr> &dtype) {
  auto output = InferOutput<1>(prim::kPrimProdExt, input_tensor, axis, keep_dims, dtype);

  int64_t axis_imm = 0;
  bool keep_dims_imm = false;
  bool is_all_reduce = false;
  if (axis.has_value()) {
    axis_imm = GetValue<int64_t>(axis.value());
    keep_dims_imm = GetValue<bool>(keep_dims);
  } else {
    is_all_reduce = true;
  }

  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = output->data_type();

  TensorToDevice(input_tensor);
  TensorMalloc(output);
  if (is_all_reduce) {
    DISPATCH_ACLNN(aclnnProd, input_tensor, out_dtype, output);
  } else {
    DISPATCH_ACLNN(aclnnProdDim, input_tensor, axis_imm, keep_dims_imm, out_dtype, output);
  }
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
