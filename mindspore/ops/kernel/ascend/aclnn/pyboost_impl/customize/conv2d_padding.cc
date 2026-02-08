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

#include "kernel/ascend/aclnn/pyboost_impl/customize/conv2d_padding.h"
#include <memory>
#include <algorithm>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim);
bool Conv2DPaddingBatchify(const ShapeVector &input_shape, const int64_t num_spatial_dims,
                           const std::string &func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  auto origin_shape_dim = SizeToLong(input_shape.size());
  const auto is_batched = (origin_shape_dim == dim_count_batch);
  if (origin_shape_dim != dim_count_no_batch && !is_batched) {
    MS_LOG(EXCEPTION) << "Expected " << dim_count_no_batch << "D (unbatched) or " << dim_count_batch
                      << "D (batched) input to " << func_name << ", but got input of size: " << origin_shape_dim;
  }
  return is_batched;
}

bool GetSymmetricPadding2D(std::vector<int64_t> &padding_l, std::vector<int64_t> &padding_r,
                           const std::vector<int64_t> &stride_vector, const std::vector<int64_t> &dilation_vector,
                           const ShapeVector &input_sizes, const ShapeVector &weight_sizes, const size_t dim) {
  bool symmetric_padding = true;
  for (size_t i = 0; i < dim; ++i) {
    auto stride_value = stride_vector.size() == 1 ? stride_vector[0] : stride_vector[i];
    auto dilation_value = dilation_vector.size() == 1 ? dilation_vector[0] : dilation_vector[i];
    auto inputSize = input_sizes[i + 2];
    auto kernelSize = weight_sizes[i + 2];
    auto total_padding = dilation_value * (kernelSize - 1);
    if (stride_value > 2 && (total_padding % 2 == 1)) {
      auto wiggle_room = inputSize % stride_value - 1;
      if (wiggle_room > 0) {
        --total_padding;
      }
    }
    auto left = total_padding / 2;
    auto right = total_padding - left;

    padding_l.push_back(left);
    padding_r.push_back(right);
    if (left != right) {
      symmetric_padding = false;
    }
  }
  return symmetric_padding;
}
}  // namespace

tensor::TensorPtr Conv2DPaddingAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                               const TensorPtr &weight_tensor,
                                               const std::optional<TensorPtr> &bias_tensor, const ValueTuplePtr &stride,
                                               const Int64ImmPtr &padding_enum, const ValueTuplePtr &dilation,
                                               const Int64ImmPtr &group) {
  // Convert ValueTuple to std::vector
  const auto &weight_shape = weight_tensor->shape();
  auto spatial_len = weight_shape.size() - kIndex2;
  std::vector<int64_t> stride_vector = ConvertValueTupleToVector<int64_t>(stride);
  ExpandParamIfNeeded(&stride_vector, spatial_len);
  std::vector<int64_t> dilation_vector = ConvertValueTupleToVector<int64_t>(dilation);
  ExpandParamIfNeeded(&dilation_vector, spatial_len);
  // Convert ValuePtr to c++ scalar
  auto padding_enum_imm = GetValue<int64_t>(padding_enum);
  auto input_shape = input_tensor->shape();
  auto is_batchify = Conv2DPaddingBatchify(input_shape, 2, "conv2d");
  TensorPtr input_tensor_new = input_tensor;
  TensorPtr expand_input_x_imm = input_tensor;
  if (!is_batchify) {
    std::vector<int64_t> expand_input_shape;
    expand_input_shape.insert(expand_input_shape.begin(), 1);
    std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(expand_input_shape),
                   [](int64_t e) { return e; });
    expand_input_x_imm = reshape(input_tensor, expand_input_shape);
    input_tensor_new = expand_input_x_imm;
  }
  std::vector<int64_t> pad_vector = {0, 0};
  if (padding_enum_imm == PadMode::SAME) {
    auto k = weight_tensor->DataNDim();
    auto dim = static_cast<size_t>(k - 2);
    auto weight_sizes = weight_tensor->shape();
    auto input_sizes = input_tensor->shape();
    if (!is_batchify) {
      input_sizes = expand_input_x_imm->shape();
    }
    std::vector<int64_t> padding_l;
    std::vector<int64_t> padding_r;
    bool symmetric_padding =
      GetSymmetricPadding2D(padding_l, padding_r, stride_vector, dilation_vector, input_sizes, weight_sizes, dim);
    if (symmetric_padding) {
      pad_vector = padding_l;
    } else {
      std::vector<ValuePtr> pad_nd(2 * dim, std::make_shared<Int64Imm>(0));
      for (size_t i = 0; i < dim; ++i) {
        // Apply padding by the difference, leaving only a symmetric padding
        auto delta_pad = padding_r[i] - padding_l[i];
        auto pad_idx = 2 * (dim - 1 - i);  // F.pad goes from last dim to first
        if (delta_pad > 0) {
          pad_nd[pad_idx + 1] = std::make_shared<Int64Imm>(delta_pad);
        } else {
          pad_nd[pad_idx] = std::make_shared<Int64Imm>(delta_pad);
          padding_l[i] = padding_r[i];
        }
      }
      auto zero = std::make_shared<Int64Imm>(0);
      if (is_batchify) {
        input_tensor_new = constant_pad_nd(input_tensor, std::make_shared<ValueTuple>(pad_nd), zero);
      } else {
        input_tensor_new = constant_pad_nd(expand_input_x_imm, std::make_shared<ValueTuple>(pad_nd), zero);
      }

      pad_vector = padding_l;
    }
  } else if (padding_enum_imm == PadMode::VALID) {
    pad_vector = {0, 0};
  } else {
    MS_LOG(EXCEPTION) << "Input padding string must be one of {'same', 'valid'}";
  }

  BoolImmPtr transposed_imm_ptr = std::make_shared<BoolImm>(false);
  ValueTuplePtr output_padding_vector_2d_imm_ptr =
    std::make_shared<ValueTuple>(std::vector<ValuePtr>({std::make_shared<Int64Imm>(0), std::make_shared<Int64Imm>(0)}));

  std::vector<ValuePtr> pad_value_ptr;
  for (int64_t i = 0; i < SizeToLong(pad_vector.size()); i++) {
    pad_value_ptr.emplace_back(std::make_shared<Int64Imm>(pad_vector[i]));
  }
  ValueTuplePtr pad_ptr = std::make_shared<ValueTuple>(pad_value_ptr);

  if (is_batchify) {
    auto output_imm = convolution(input_tensor_new, weight_tensor, bias_tensor, stride, pad_ptr, dilation,
                                  transposed_imm_ptr, output_padding_vector_2d_imm_ptr, group);
    op->set_outputs({output_imm});
    return output_imm;
  } else {
    auto output_imm = convolution(input_tensor_new, weight_tensor, bias_tensor, stride, pad_ptr, dilation,
                                  transposed_imm_ptr, output_padding_vector_2d_imm_ptr, group);
    auto output_imm_shape = output_imm->shape();
    std::vector<int64_t> squeeze_output_shape;
    for (int64_t i = 1; i < SizeToLong(output_imm_shape.size()); i++) {
      squeeze_output_shape.emplace_back(output_imm_shape[i]);
    }
    auto squeeze_output_tensor = reshape(output_imm, squeeze_output_shape);
    op->set_outputs({squeeze_output_tensor});
    return squeeze_output_tensor;
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
