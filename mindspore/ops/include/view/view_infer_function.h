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

#ifndef MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_FUNCTION_H
#define MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_FUNCTION_H

#include <vector>
#include "view/view_infer_reg.h"
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace ops {
OPS_API size_t FetchChunkOutputNum(const tensor::TensorPtr &input, const int64_t &chunks, const int64_t &dim);
REG_VIEW_INFER_FUNCTION(Chunk, FetchChunkOutputNum);
REG_VIEW_INFER_FUNCTION(ChunkView, FetchChunkOutputNum);

OPS_API size_t FetchSplitOutputNum(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &axis,
                                   const int64_t &output_num);
REG_VIEW_INFER_FUNCTION(Split, FetchSplitOutputNum);

OPS_API size_t FetchSplitTensorOutputNum(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &split_size,
                                         const int64_t &dim);
REG_VIEW_INFER_FUNCTION(SplitTensor, FetchSplitTensorOutputNum);
REG_VIEW_INFER_FUNCTION(SplitTensorView, FetchSplitTensorOutputNum);

OPS_API size_t FetchSplitWithSizeOutputNum(const mindspore::tensor::TensorPtr &input_tensor,
                                           const std::vector<int64_t> &split_size, const int64_t &dim);
REG_VIEW_INFER_FUNCTION(SplitWithSize, FetchSplitWithSizeOutputNum);
REG_VIEW_INFER_FUNCTION(SplitWithSizeView, FetchSplitWithSizeOutputNum);

OPS_API size_t FetchUnstackExtViewOutputNum(const tensor::TensorPtr &input, const int64_t &dim);
REG_VIEW_INFER_FUNCTION(UnstackExtView, FetchUnstackExtViewOutputNum);

OPS_API size_t FetchMeshgridOutputNum(const ValueTuplePtr &inputs, const int64_t &indexing);
REG_VIEW_INFER_FUNCTION(Meshgrid, FetchMeshgridOutputNum);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_OPS_INCLUDE_VIEW_VIEW_INFER_FUNCTION_H
