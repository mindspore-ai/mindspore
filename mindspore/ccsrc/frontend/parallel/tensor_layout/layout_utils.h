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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_UTILS_H_
#include <map>
#include <vector>
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore::parallel {
Status GetFactors(const TensorLayout &layout, Array *array);
void InitShapeVec(const Shape &src_shape, Shape *tgt_shape);
int64_t GetTensorSize(const Shape &shape);
int64_t GetLeastFactorWithoutConstDims(const Shape &to_shape, const Array &to_factors);
bool UseStrictMode(const Shape &from_shape, const Shape &to_shape);
bool RecordDimsChange(size_t key, int64_t value, std::map<size_t, int64_t> *memo, bool update = false);
void IntroduceConstraints(const Shape &expected_tgt_shape, Shape *tgt_shape);
bool ForwardMatching(const Shape &src_shape, const Shape &expected_tgt_shape, Shape *tgt_shape,
                     const Array &tgt_factors);
bool BackwardMatching(const Shape &expected_tgt_shape, Shape *tgt_shape, const Array &tgt_factors);
bool CheckDynamicShape(const TensorLayout &from_in, const TensorLayout &to_in);
bool SolveCombination(const Shape &src_shape_arr, size_t src_index,
                      const std::vector<std::vector<int64_t>> &enum_numbers, size_t offset, int64_t target,
                      std::vector<int64_t> *candidates_values);
void UnifyFromAndToShape(Shape *new_from_shape, Shape *new_to_shape, const TensorLayout &from_in,
                         const TensorLayout &to_in, ReplacementMemo *from_dims_replace_memo);
}  // namespace mindspore::parallel
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_UTILS_H_
