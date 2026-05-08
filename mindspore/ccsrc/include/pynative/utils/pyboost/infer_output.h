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

#ifndef MINDSPORE_CCSRC_KERNEL_PYBOOST_INFER_OUTPUT_H_
#define MINDSPORE_CCSRC_KERNEL_PYBOOST_INFER_OUTPUT_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "include/backend/visible.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/infer_info/infer_info_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace infer_detail {
// ---------- Return type traits ----------
// Generate std::tuple<TensorPtr, TensorPtr, ...> with N elements
template <typename T, size_t N, typename = std::make_index_sequence<N>>
struct RepeatTuple;

template <typename T, size_t N, size_t... Is>
struct RepeatTuple<T, N, std::index_sequence<Is...>> {
  template <size_t>
  using Elem = T;
  using type = std::tuple<Elem<Is>...>;
};

// InferOutputResult<N>:
//   N == 0  => std::vector<TensorPtr>   (dynamic output count, e.g., split)
//   N == 1  => TensorPtr                (single output, e.g., matmul_ext)
//   N >  1  => std::tuple<TensorPtr...> (fixed multi-output, e.g., batch_norm_ext)
template <size_t N>
using InferOutputResult =
  std::conditional_t<N == 0, std::vector<tensor::TensorPtr>,
                     std::conditional_t<N == 1, tensor::TensorPtr, typename RepeatTuple<tensor::TensorPtr, N>::type>>;

// ---------- Output packing (struct specialization to avoid if constexpr) ----------
template <size_t N>
struct OutputPacker {
  template <size_t... I>
  static typename RepeatTuple<tensor::TensorPtr, N>::type DoPackImpl(const std::vector<tensor::TensorPtr> &outputs,
                                                                     std::index_sequence<I...>) {
    return std::make_tuple(outputs[I]...);
  }
  static InferOutputResult<N> Pack(const std::vector<tensor::TensorPtr> &outputs) {
    return DoPackImpl(outputs, std::make_index_sequence<N>{});
  }
};

template <>
struct OutputPacker<0> {
  static std::vector<tensor::TensorPtr> Pack(const std::vector<tensor::TensorPtr> &outputs) { return outputs; }
};

template <>
struct OutputPacker<1> {
  static tensor::TensorPtr Pack(const std::vector<tensor::TensorPtr> &outputs) { return outputs[0]; }
};

// ---------- Core inference logic ----------
template <typename... T>
std::vector<tensor::TensorPtr> DoInfer(const PrimitivePtr &primitive, const T &...args) {
  std::vector<tensor::TensorPtr> outputs;

  // Path 1: DoGeneralInfer (fastest path, uses InferInfo)
  auto value_simple_info = ops::DoGeneralInfer(primitive, args...);
  if (value_simple_info != nullptr) {
    MS_LOG(DEBUG) << "Op " << primitive->name() << " infer by infer_info, get output "
                  << ValueSimpleInfoToString(*value_simple_info);
    PyBoostUtils::CreateOutputTensor(value_simple_info, &outputs);
    return outputs;
  }

  // Path 2: InferBySimple (uses SimpleInfer registry)
  value_simple_info = ops::InferBySimple(primitive, args...);
  if (value_simple_info != nullptr) {
    MS_LOG(DEBUG) << "Op " << primitive->name() << " infer by simple, get output "
                  << ValueSimpleInfoToString(*value_simple_info);
    PyBoostUtils::CreateOutputTensor(value_simple_info, &outputs);
    return outputs;
  }

  // Path 3: Fallback to abstract-based inference
  static pynative::AbstractConverter abstract_converter;
  std::vector<AbstractBasePtr> input_abs;
  (input_abs.emplace_back(abstract_converter.ConvertAbstract(args)), ...);
  auto output_abs = PyBoostUtils::InferByOpDef(primitive, input_abs);
  MS_EXCEPTION_IF_NULL(output_abs);
  MS_LOG(DEBUG) << "PyBoost infer by abstract, get output " << output_abs->ToString();
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  abstract_converter.CacheAbstract(output_abs);
  return outputs;
}
}  // namespace infer_detail

/// \brief Perform shape/dtype inference and create output tensors, decoupled from OpRunner.
///
/// Template parameter N is the number of outputs defined in the op's YAML:
///   N == 0 (dynamic, e.g., split)     => returns std::vector<TensorPtr>
///   N == 1 (e.g., matmul_ext)         => returns TensorPtr
///   N >  1 (e.g., batch_norm_ext: 3)  => returns std::tuple<TensorPtr, ...>
///
/// Inference follows the same 3-path strategy as OpRunner::InferOutput:
///   Path 1 (fastest):  ops::DoGeneralInfer
///   Path 2:            ops::InferBySimple
///   Path 3 (fallback): abstract-based inference via PyBoostUtils::InferByOpDef
///
/// \param primitive A static PrimitivePtr for the op
/// \param args The original input arguments (TensorPtr, ScalarPtr, optional<TensorPtr>, etc.)
/// \return See N description above.
template <size_t N, typename... T>
infer_detail::InferOutputResult<N> InferOutput(const PrimitivePtr &primitive, const T &...args) {
  auto outputs = infer_detail::DoInfer(primitive, args...);
  return infer_detail::OutputPacker<N>::Pack(outputs);
}

/// \brief Perform shape/dtype inference for ops with dynamic output count (e.g., tuple[tensor]).
template <typename... T>
std::vector<tensor::TensorPtr> InferDynamicOutput(const PrimitivePtr &primitive, const T &...args) {
  return InferOutput<0>(primitive, args...);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_PYBOOST_INFER_OUTPUT_H_
