/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/nn_batch_norm_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// BatchNorm
INPUT_MAP(BatchNorm) = {{1, INPUT_DESC(x)},
                        {2, INPUT_DESC(scale)},
                        {3, INPUT_DESC(offset)},
                        {4, INPUT_DESC(mean)},
                        {5, INPUT_DESC(variance)}};
ATTR_MAP(BatchNorm) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                       {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNorm) = {{0, OUTPUT_DESC(y)},
                         {1, OUTPUT_DESC(batch_mean)},
                         {2, OUTPUT_DESC(batch_variance)},
                         {3, OUTPUT_DESC(reserve_space_1)},
                         {4, OUTPUT_DESC(reserve_space_2)}};
REG_ADPT_DESC(BatchNorm, kNameBatchNorm, ADPT_DESC(BatchNorm))

// BatchNormGrad
INPUT_MAP(BatchNormGrad) = {{1, INPUT_DESC(y_backprop)},
                            {2, INPUT_DESC(x)},
                            {3, INPUT_DESC(scale)},
                            {4, INPUT_DESC(reserve_space_1)},
                            {5, INPUT_DESC(reserve_space_2)}};
ATTR_MAP(BatchNormGrad) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                           {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                           {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNormGrad) = {{0, OUTPUT_DESC(x_backprop)},
                             {1, OUTPUT_DESC(scale_backprop)},
                             {2, OUTPUT_DESC(offset_backprop)},
                             {3, OUTPUT_DESC(reserve_space_4)},
                             {4, OUTPUT_DESC(reserve_space_5)}};
REG_ADPT_DESC(BatchNormGrad, kNameBatchNormGrad, ADPT_DESC(BatchNormGrad))

// L2NormalizeGrad
INPUT_MAP(L2NormalizeGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(dy)}};
ATTR_MAP(L2NormalizeGrad) = {
  {"axis", ATTR_DESC(dim, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2NormalizeGrad) = {{0, OUTPUT_DESC(dx)}};
REG_ADPT_DESC(L2NormalizeGrad, kNameL2NormalizeGrad, ADPT_DESC(L2NormalizeGrad))

// L2Normalize
INPUT_MAP(L2Normalize) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Normalize) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2Normalize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(L2Normalize, kNameL2Normalize, ADPT_DESC(L2Normalize))
}  // namespace mindspore::transform
