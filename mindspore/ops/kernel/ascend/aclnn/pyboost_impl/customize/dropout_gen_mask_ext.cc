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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dropout_gen_mask_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DropoutGenMaskExtAscendCustomize(const ValueTuplePtr &shape, const FP32ImmPtr &p,
                                                   const TensorPtr &seed, const TensorPtr &offset,
                                                   const Int64ImmPtr &dtype) {
  auto output = InferOutput<1>(prim::kPrimDropoutGenMaskExt, shape, p, seed, offset, dtype);

  auto shape_vector = ConvertValueTupleToVector<int64_t>(shape);
  auto [seed_value, offset_value] = GetGeneratorState(seed, offset);
  auto p_value = static_cast<double>(p->value());
  auto dtype_value = static_cast<TypeId>(dtype->value());
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnDropoutGenMaskV2, shape_vector, p_value, seed_value, offset_value, dtype_value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
