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

#include "kernel/ascend/aclnn/pyboost_impl/customize/convolution_str.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/constant_pad_nd.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim) {
  if (param->size() == kIndex1) {
    param->insert(param->end(), expect_dim - kIndex1, param->at(kIndex0));
  }
}
}  // namespace
tensor::TensorPtr ConvolutionStrAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &weight_tensor,
                                                const std::optional<TensorPtr> &bias_tensor,
                                                const ValueTuplePtr &stride, const Int64ImmPtr &padding_enum,
                                                const ValueTuplePtr &dilation, const BoolImmPtr &transposed,
                                                const ValueTuplePtr &output_padding, const Int64ImmPtr &group) {
  auto output = InferOutput<1>(prim::kPrimConvolutionStr, input_tensor, weight_tensor, bias_tensor, stride,
                               padding_enum, dilation, transposed, output_padding, group);
  // Convert ValueTuple to std::vector
  const auto &weight_shape = weight_tensor->shape();
  auto spatial_len = weight_shape.size() - kIndex2;
  std::vector<int64_t> stride_vector = ConvertValueTupleToVector<int64_t>(stride);
  ExpandParamIfNeeded(&stride_vector, spatial_len);
  std::vector<int64_t> dilation_vector = ConvertValueTupleToVector<int64_t>(dilation);
  ExpandParamIfNeeded(&dilation_vector, spatial_len);
  std::vector<int64_t> output_padding_vector = ConvertValueTupleToVector<int64_t>(output_padding);
  ExpandParamIfNeeded(&output_padding_vector, spatial_len);
  // Convert ValuePtr to c++ scalar
  auto transposed_imm = GetValue<bool>(transposed);
  auto group_imm = GetValue<int64_t>(group);
  auto padding_enum_imm = GetValue<int64_t>(padding_enum);

  TensorPtr input_tensor_new = input_tensor;
  auto k = weight_tensor->DataNDim();
  auto dim = static_cast<size_t>(k - 2);
  std::vector<int64_t> pad_vector = std::vector<int64_t>(dim, 0);
  if (padding_enum_imm == PadMode::SAME) {
    auto weight_sizes = weight_tensor->shape();
    auto input_sizes = input_tensor->shape();

    std::vector<int64_t> padding_l;
    std::vector<int64_t> padding_r;
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
    if (symmetric_padding) {
      MS_LOG(INFO) << "ConvolutionStr: symmetric padding is True.";
      pad_vector = padding_l;
    } else {
      MS_LOG(INFO) << "ConvolutionStr: symmetric padding is False.";
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
      auto constant_pad_nd_op = CREATE_PYBOOST_OP(ConstantPadND, device::DeviceType::kAscend);
      MS_LOG(INFO) << "ConvolutionStr: pad_nd is " << pad_nd;
      input_tensor_new =
        constant_pad_nd_op->Call(input_tensor, std::make_shared<ValueTuple>(pad_nd), zero);  // 注意是否可用

      pad_vector = padding_l;
    }
  } else if (padding_enum_imm == PadMode::VALID) {
    MS_LOG(INFO) << "For Primitive[ConvolutionStr], paddingmode is value.";
  } else {
    MS_LOG(EXCEPTION) << "Input padding string must be one of {'same', 'valid'}";
  }

  TensorToDevice(input_tensor_new, weight_tensor, bias_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnConvolution, input_tensor_new, weight_tensor, bias_tensor, stride_vector, pad_vector,
                 dilation_vector, transposed_imm, output_padding_vector, group_imm, output, GetCubeMathType());
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
