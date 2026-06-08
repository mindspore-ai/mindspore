/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kIntMax = 2147483647;
constexpr int64_t kSoftmaxLastDim = 8;

void CheckRank(const ms::Tensor &tensor, size_t rank, const std::string &name) {
  if (tensor.shape().size() != rank) {
    throw std::runtime_error(name + " rank should be " + std::to_string(rank));
  }
}

ShapeVector BatchMatMulShape(const ShapeVector &x_shape, const ShapeVector &y_shape) {
  if (x_shape.size() < 2 || y_shape.size() < 2) {
    throw std::runtime_error("matmul inputs should have rank >= 2");
  }

  const size_t x_batch_rank = x_shape.size() - 2;
  const size_t y_batch_rank = y_shape.size() - 2;
  const size_t batch_rank = std::max(x_batch_rank, y_batch_rank);
  ShapeVector output_shape(batch_rank + 2);

  for (size_t i = 0; i < batch_rank; ++i) {
    const int64_t x_dim = i < batch_rank - x_batch_rank ? 1 : x_shape[i - (batch_rank - x_batch_rank)];
    const int64_t y_dim = i < batch_rank - y_batch_rank ? 1 : y_shape[i - (batch_rank - y_batch_rank)];
    if (x_dim != y_dim && x_dim != 1 && y_dim != 1) {
      throw std::runtime_error("matmul batch dimensions cannot be broadcast");
    }
    output_shape[i] = std::max(x_dim, y_dim);
  }

  if (x_shape[x_shape.size() - 1] != y_shape[y_shape.size() - 2]) {
    throw std::runtime_error("matmul contracting dimensions do not match");
  }
  output_shape[batch_rank] = x_shape[x_shape.size() - 2];
  output_shape[batch_rank + 1] = y_shape[y_shape.size() - 1];
  return output_shape;
}

ShapeVector LayerNormStatShape(const ShapeVector &x_shape, const std::vector<int64_t> &normalized_shape) {
  if (normalized_shape.empty() || normalized_shape.size() > x_shape.size()) {
    throw std::runtime_error("invalid normalized_shape for layer_norm");
  }
  const size_t normalized_rank = normalized_shape.size();
  const size_t outer_rank = x_shape.size() - normalized_rank;
  ShapeVector stat_shape(x_shape.begin(), x_shape.begin() + static_cast<int64_t>(outer_rank));
  for (size_t i = 0; i < normalized_rank; ++i) {
    if (x_shape[outer_rank + i] != normalized_shape[i]) {
      throw std::runtime_error("normalized_shape should match input trailing dimensions");
    }
    stat_shape.push_back(1);
  }
  return stat_shape;
}
}  // namespace

ms::Tensor npu_llm_matmul_v2(const ms::Tensor &x, const ms::Tensor &y) {
  auto result = ms::Tensor(x.data_type(), BatchMatMulShape(x.shape(), y.shape()));
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(result);
  MS_DISPATCH_ACLNN(aclnnMatmul, x, y, result, static_cast<int8_t>(0));
  return result;
}

ms::Tensor npu_llm_layer_norm_v2(const ms::Tensor &x, const std::vector<int64_t> &normalized_shape,
                                 const ms::Tensor &weight, const ms::Tensor &bias, double eps) {
  const auto stat_shape = LayerNormStatShape(x.shape(), normalized_shape);
  std::vector<ms::Tensor> outputs{ms::Tensor(x.data_type(), x.shape()),
                                  ms::Tensor(ms::TypeId::kNumberTypeFloat32, stat_shape),
                                  ms::Tensor(ms::TypeId::kNumberTypeFloat32, stat_shape)};
  ms::TensorToDevice(x, weight, bias);
  ms::TensorAllocate(outputs);
  MS_DISPATCH_ACLNN(aclnnLayerNorm, x, normalized_shape, weight, bias, eps, outputs[0], outputs[1], outputs[2]);
  return outputs[0];
}

ms::Tensor npu_llm_flash_attention_v2(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                                      double scale_value) {
  CheckRank(query, 4, "query");
  CheckRank(key, 4, "key");
  CheckRank(value, 4, "value");
  const auto &query_shape = query.shape();
  const auto &key_shape = key.shape();
  const auto &value_shape = value.shape();
  if (query_shape[0] != key_shape[0] || query_shape[0] != value_shape[0] || query_shape[1] != key_shape[1] ||
      query_shape[1] != value_shape[1] || query_shape[3] != key_shape[3] || key_shape[2] != value_shape[2]) {
    throw std::runtime_error("flash attention inputs should use compatible BNSD shapes");
  }

  ShapeVector softmax_shape{query_shape[0], query_shape[1], query_shape[2], kSoftmaxLastDim};
  ShapeVector output_shape{query_shape[0], query_shape[1], query_shape[2], value_shape[3]};
  std::vector<ms::Tensor> outputs{ms::Tensor(ms::TypeId::kNumberTypeFloat32, softmax_shape),
                                  ms::Tensor(ms::TypeId::kNumberTypeFloat32, softmax_shape),
                                  ms::Tensor(query.data_type(), ShapeVector{1}),
                                  ms::Tensor(query.data_type(), output_shape)};

  std::optional<ms::Tensor> real_shift = std::nullopt;
  std::optional<ms::Tensor> drop_mask = std::nullopt;
  std::optional<ms::Tensor> padding_mask = std::nullopt;
  std::optional<ms::Tensor> attn_mask = std::nullopt;
  std::vector<int64_t> prefix;
  const double keep_prob = 1.0;
  const int64_t pre_tokens = kIntMax;
  const int64_t next_tokens = kIntMax;
  const int64_t head_num = query_shape[1];
  const std::string input_layout = "BNSD";
  const int64_t inner_precise = 0;
  const int64_t sparse_mode = 0;

  ms::TensorToDevice(query, key, value, real_shift, drop_mask, padding_mask, attn_mask);
  ms::TensorAllocate(outputs);
  MS_DISPATCH_ACLNN(aclnnFlashAttentionScore, query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix,
                    scale_value, keep_prob, pre_tokens, next_tokens, head_num, input_layout, inner_precise, sparse_mode,
                    outputs[0], outputs[1], outputs[2], outputs[3]);
  return outputs[3];
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_llm_matmul_v2", PYBOOST_CALLER(1, custom::npu_llm_matmul_v2));
  m.def("npu_llm_layer_norm_v2", PYBOOST_CALLER(1, custom::npu_llm_layer_norm_v2));
  m.def("npu_llm_flash_attention_v2", PYBOOST_CALLER(1, custom::npu_llm_flash_attention_v2));
}
