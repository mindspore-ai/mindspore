/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/math_ops_declare.h"

namespace mindspore::transform {
// NLLLoss
INPUT_MAP(NLLLoss) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(weight)}};
ATTR_MAP(NLLLoss) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(NLLLoss) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(total_weight)}};
REG_ADPT_DESC(NLLLoss, kNameNLLLoss, ADPT_DESC(NLLLoss))

// NLLLossGrad
INPUT_MAP(NLLLossGrad) = {{1, INPUT_DESC(x)},
                          {2, INPUT_DESC(y_grad)},
                          {3, INPUT_DESC(target)},
                          {4, INPUT_DESC(weight)},
                          {5, INPUT_DESC(total_weight)}};
ATTR_MAP(NLLLossGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(NLLLossGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(NLLLossGrad, kNameNLLLossGrad, ADPT_DESC(NLLLossGrad))

// Erf
INPUT_MAP(Erf) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Erf) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Erf) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Erf, kNameErf, ADPT_DESC(Erf))

// Erfc
INPUT_MAP(Erfc) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Erfc) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Erfc) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Erfc, kNameErfc, ADPT_DESC(Erfc))

// WtsARQ
INPUT_MAP(WtsARQ) = {{1, INPUT_DESC(w)}, {2, INPUT_DESC(w_min)}, {3, INPUT_DESC(w_max)}};
ATTR_MAP(WtsARQ) = {{"num_bits", ATTR_DESC(num_bits, AnyTraits<int64_t>())},
                    {"offset_flag", ATTR_DESC(offset_flag, AnyTraits<bool>())}};
OUTPUT_MAP(WtsARQ) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(WtsARQ, kNameWtsARQ, ADPT_DESC(WtsARQ))
}  // namespace mindspore::transform
