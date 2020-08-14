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

#include "transform/graph_ir/op_declare/rnn_declare.h"

namespace mindspore::transform {
// BasicLSTMCell
INPUT_MAP(BasicLSTMCell) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(h)}, {3, INPUT_DESC(c)}, {4, INPUT_DESC(w)}, {5, INPUT_DESC(b)}};
ATTR_MAP(BasicLSTMCell) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                           {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                           {"state_is_tuple", ATTR_DESC(state_is_tuple, AnyTraits<bool>())},
                           {"activation", ATTR_DESC(activation, AnyTraits<std::string>())}};
OUTPUT_MAP(BasicLSTMCell) = {{0, OUTPUT_DESC(ct)}, {1, OUTPUT_DESC(ht)}, {2, OUTPUT_DESC(it)},    {3, OUTPUT_DESC(jt)},
                             {4, OUTPUT_DESC(ft)}, {5, OUTPUT_DESC(ot)}, {7, OUTPUT_DESC(tanhct)}};
REG_ADPT_DESC(BasicLSTMCell, kNameBasicLSTMCell, ADPT_DESC(BasicLSTMCell))

// BasicLSTMCellInputGrad
INPUT_MAP(BasicLSTMCellInputGrad) = {{1, INPUT_DESC(dgate)}, {2, INPUT_DESC(w)}};
ATTR_MAP(BasicLSTMCellInputGrad) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())}};
OUTPUT_MAP(BasicLSTMCellInputGrad) = {{0, OUTPUT_DESC(dxt)}, {1, OUTPUT_DESC(dht)}};
REG_ADPT_DESC(BasicLSTMCellInputGrad, kNameBasicLSTMCellInputGrad, ADPT_DESC(BasicLSTMCellInputGrad))

// BasicLSTMCellWeightGrad
INPUT_MAP(BasicLSTMCellWeightGrad) = {{1, INPUT_DESC(h)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(dgate)}};
ATTR_MAP(BasicLSTMCellWeightGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BasicLSTMCellWeightGrad) = {{0, OUTPUT_DESC(dw)}, {1, OUTPUT_DESC(db)}};
REG_ADPT_DESC(BasicLSTMCellWeightGrad, kNameBasicLSTMCellWeightGrad, ADPT_DESC(BasicLSTMCellWeightGrad))

// BasicLSTMCellCStateGrad
INPUT_MAP(BasicLSTMCellCStateGrad) = {{1, INPUT_DESC(c)},  {2, INPUT_DESC(dht)},   {3, INPUT_DESC(dct)},
                                      {4, INPUT_DESC(it)}, {5, INPUT_DESC(jt)},    {6, INPUT_DESC(ft)},
                                      {7, INPUT_DESC(ot)}, {8, INPUT_DESC(tanhct)}};
ATTR_MAP(BasicLSTMCellCStateGrad) = {{"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                                     {"activation", ATTR_DESC(activation, AnyTraits<std::string>())}};
OUTPUT_MAP(BasicLSTMCellCStateGrad) = {{0, OUTPUT_DESC(dgate)}, {1, OUTPUT_DESC(dct_1)}};
REG_ADPT_DESC(BasicLSTMCellCStateGrad, kNameBasicLSTMCellCStateGrad, ADPT_DESC(BasicLSTMCellCStateGrad))
}  // namespace mindspore::transform
