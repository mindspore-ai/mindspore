/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/reduce_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// BNTrainingReduce
INPUT_MAP(BNTrainingReduce) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BNTrainingReduce) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BNTrainingReduce) = {{0, OUTPUT_DESC(sum)}, {1, OUTPUT_DESC(square_sum)}};
REG_ADPT_DESC(BNTrainingReduce, kNameBNTrainingReduce, ADPT_DESC(BNTrainingReduce))

// BNTrainingReduceGrad
INPUT_MAP(BNTrainingReduceGrad) = {{1, INPUT_DESC(grads)},         {2, INPUT_DESC(x)},     {3, INPUT_DESC(diff_scale)},
                                   {4, INPUT_DESC(diff_offset)},   {5, INPUT_DESC(scale)}, {6, INPUT_DESC(batch_mean)},
                                   {7, INPUT_DESC(batch_variance)}};
ATTR_MAP(BNTrainingReduceGrad) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(BNTrainingReduceGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BNTrainingReduceGrad, kNameBNTrainingReduceGrad, ADPT_DESC(BNTrainingReduceGrad))

// BNTrainingUpdate
INPUT_MAP(BNTrainingUpdate) = {{1, INPUT_DESC(x)},       {2, INPUT_DESC(sum)},    {3, INPUT_DESC(square_sum)},
                               {4, INPUT_DESC(scale)},   {5, INPUT_DESC(offset)}, {6, INPUT_DESC(mean)},
                               {7, INPUT_DESC(variance)}};
ATTR_MAP(BNTrainingUpdate) = {{"factor", ATTR_DESC(factor, AnyTraits<float>())},
                              {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(BNTrainingUpdate) = {{0, OUTPUT_DESC(y)},
                                {1, OUTPUT_DESC(mean)},
                                {2, OUTPUT_DESC(variance)},
                                {3, OUTPUT_DESC(batch_mean)},
                                {4, OUTPUT_DESC(batch_variance)}};
REG_ADPT_DESC(BNTrainingUpdate, kNameBNTrainingUpdate, ADPT_DESC(BNTrainingUpdate))

// BNTrainingUpdateGrad
INPUT_MAP(BNTrainingUpdateGrad) = {
  {1, INPUT_DESC(grads)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(batch_mean)}, {4, INPUT_DESC(batch_variance)}};
ATTR_MAP(BNTrainingUpdateGrad) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(BNTrainingUpdateGrad) = {{0, OUTPUT_DESC(diff_scale)}, {1, OUTPUT_DESC(diff_offset)}};
REG_ADPT_DESC(BNTrainingUpdateGrad, kNameBNTrainingUpdateGrad, ADPT_DESC(BNTrainingUpdateGrad))

// ReduceAny
INPUT_MAP(ReduceAny) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceAny) = {{"axis", 2}};
ATTR_MAP(ReduceAny) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceAny) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceAny, prim::kPrimReduceAny->name(), ADPT_DESC(ReduceAny))

// ReduceSumD
INPUT_MAP(ReduceSumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceSumD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceSumD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceSumD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceSumD, prim::kPrimReduceSum->name(), ADPT_DESC(ReduceSumD))

// ReduceSum
INPUT_MAP(ReduceSum) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceSum) = {{"axis", 2}};
ATTR_MAP(ReduceSum) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceSum) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceSum, "ReduceSumD", ADPT_DESC(ReduceSum))

// ReduceProd
INPUT_MAP(ReduceProd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceProd) = {{"axis", 2}};
ATTR_MAP(ReduceProd) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceProd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceProd, kNameReduceProd, ADPT_DESC(ReduceProd))

// ReduceAll
INPUT_MAP(ReduceAll) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceAll) = {{"axis", 2}};
ATTR_MAP(ReduceAll) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceAll) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceAll, prim::kPrimReduceAll->name(), ADPT_DESC(ReduceAll))

// ReduceMean
INPUT_MAP(ReduceMean) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceMean) = {{"axis", 2}};
ATTR_MAP(ReduceMean) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())},
                        {"noop_with_empty_axes", ATTR_DESC(noop_with_empty_axes, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMean) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMean, prim::kPrimReduceMean->name(), ADPT_DESC(ReduceMean))

// ReduceMin
INPUT_MAP(ReduceMin) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceMin) = {{"axis", 2}};
ATTR_MAP(ReduceMin) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMin) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMin, prim::kPrimReduceMin->name(), ADPT_DESC(ReduceMin))

// ReduceMax
INPUT_MAP(ReduceMax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_INPUT_MAP(ReduceMax) = {{"axis", 2}};
ATTR_MAP(ReduceMax) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMax) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMax, prim::kPrimReduceMax->name(), ADPT_DESC(ReduceMax))
}  // namespace mindspore::transform
