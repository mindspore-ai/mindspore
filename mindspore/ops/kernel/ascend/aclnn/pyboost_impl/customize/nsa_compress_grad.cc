/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_compress_grad.h"
#include <string>
#include <vector>
#include <utility>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> NsaCompressGradAscendCustomize(
  const TensorPtr &grad, const TensorPtr &input, const TensorPtr &weight, const Int64ImmPtr &compress_block_size,
  const Int64ImmPtr &compress_stride, const ValueTuplePtr &actual_seq_len) {
  auto outputs = InferOutput<2>(prim::kPrimNsaCompressGrad, grad, input, weight, compress_block_size, compress_stride,
                                actual_seq_len);

  auto compress_block_size_value = GetValue<int64_t>(compress_block_size);
  auto compress_stride_value = GetValue<int64_t>(compress_stride);
  std::vector<int64_t> actual_seq_len_array;
  actual_seq_len_array = ConvertValueTupleToVector<int64_t>(actual_seq_len);
  const std::string layout_string = "TND";
  const int64_t actual_seq_len_type_value = 0;
  const auto actual_seq_len_pair = std::make_pair(actual_seq_len_array, true);

  TensorToDevice(grad, input, weight);
  TensorMalloc(outputs);
  DISPATCH_ACLNN(aclnnNsaCompressGrad, grad, input, weight, actual_seq_len_pair, compress_block_size_value,
                 compress_stride_value, actual_seq_len_type_value, layout_string, std::get<0>(outputs),
                 std::get<1>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
