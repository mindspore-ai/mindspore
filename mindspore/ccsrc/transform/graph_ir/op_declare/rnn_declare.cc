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
                             {4, OUTPUT_DESC(ft)}, {5, OUTPUT_DESC(ot)}, {6, OUTPUT_DESC(tanhct)}};
REG_ADPT_DESC(BasicLSTMCell, kNameBasicLSTMCell, ADPT_DESC(BasicLSTMCell))

// BasicLSTMCellInputGrad
INPUT_MAP(BasicLSTMCellInputGrad) = {{1, INPUT_DESC(dgate)}, {2, INPUT_DESC(w)}};
ATTR_MAP(BasicLSTMCellInputGrad) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())}};
OUTPUT_MAP(BasicLSTMCellInputGrad) = {{0, OUTPUT_DESC(dxt)}, {1, OUTPUT_DESC(dht)}};
REG_ADPT_DESC(BasicLSTMCellInputGrad, kNameBasicLSTMCellInputGrad, ADPT_DESC(BasicLSTMCellInputGrad))

// BasicLSTMCellWeightGrad
INPUT_MAP(BasicLSTMCellWeightGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(h)}, {3, INPUT_DESC(dgate)}};
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

// DynamicRNN
INPUT_MAP(DynamicRNN) = {{1, INPUT_DESC(x)},          {2, INPUT_DESC(w)},      {3, INPUT_DESC(b)},
                         {4, INPUT_DESC(seq_length)}, {5, INPUT_DESC(init_h)}, {6, INPUT_DESC(init_c)},
                         {7, INPUT_DESC(wci)},        {8, INPUT_DESC(wcf)},    {9, INPUT_DESC(wco)},
                         {10, INPUT_DESC(mask)}};
ATTR_MAP(DynamicRNN) = {{"cell_type", ATTR_DESC(cell_type, AnyTraits<std::string>())},
                        {"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                        {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                        {"use_peephole", ATTR_DESC(use_peephole, AnyTraits<bool>())},
                        {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                        {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                        {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                        {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                        {"ivation", ATTR_DESC(activation, AnyTraits<std::string>())},
                        {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())},
                        {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(DynamicRNN) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(output_h)}, {2, OUTPUT_DESC(output_c)},
                          {3, OUTPUT_DESC(i)}, {4, OUTPUT_DESC(j)},        {5, OUTPUT_DESC(f)},
                          {6, OUTPUT_DESC(o)}, {7, OUTPUT_DESC(tanhc)}};
REG_ADPT_DESC(DynamicRNN, kNameDynamicRNN, ADPT_DESC(DynamicRNN))

// DynamicRNNGrad
INPUT_MAP(DynamicRNNGrad) = {
  {1, INPUT_DESC(x)},      {2, INPUT_DESC(w)},      {3, INPUT_DESC(b)},   {4, INPUT_DESC(y)},
  {5, INPUT_DESC(init_h)}, {6, INPUT_DESC(init_c)}, {7, INPUT_DESC(h)},   {8, INPUT_DESC(c)},
  {9, INPUT_DESC(dy)},     {10, INPUT_DESC(dh)},    {11, INPUT_DESC(dc)}, {12, INPUT_DESC(i)},
  {13, INPUT_DESC(j)},     {14, INPUT_DESC(f)},     {15, INPUT_DESC(o)},  {16, INPUT_DESC(tanhct)}};

ATTR_MAP(DynamicRNNGrad) = {{"cell_type", ATTR_DESC(cell_type, AnyTraits<std::string>())},
                            {"direction", ATTR_DESC(direction, AnyTraits<std::string>())},
                            {"cell_depth", ATTR_DESC(cell_depth, AnyTraits<int64_t>())},
                            {"use_peephole", ATTR_DESC(use_peephole, AnyTraits<bool>())},
                            {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
                            {"cell_clip", ATTR_DESC(cell_clip, AnyTraits<float>())},
                            {"num_proj", ATTR_DESC(num_proj, AnyTraits<int64_t>())},
                            {"time_major", ATTR_DESC(time_major, AnyTraits<bool>())},
                            {"forget_bias", ATTR_DESC(forget_bias, AnyTraits<float>())}};
OUTPUT_MAP(DynamicRNNGrad) = {{0, OUTPUT_DESC(dw)},
                              {1, OUTPUT_DESC(db)},
                              {2, OUTPUT_DESC(dx)},
                              {3, OUTPUT_DESC(dh_prev)},
                              {4, OUTPUT_DESC(dc_prev)}};
REG_ADPT_DESC(DynamicRNNGrad, kNameDynamicRNNGrad, ADPT_DESC(DynamicRNNGrad))
}  // namespace mindspore::transform
