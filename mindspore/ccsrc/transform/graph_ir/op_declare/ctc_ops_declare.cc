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

#include "transform/graph_ir/op_declare/ctc_ops_declare.h"
#include <string>

namespace mindspore::transform {
// CTCLoss
INPUT_MAP(CTCLoss) = {{1, INPUT_DESC(inputs)},
                      {2, INPUT_DESC(labels_indices)},
                      {3, INPUT_DESC(labels_values)},
                      {4, INPUT_DESC(sequence_length)}};
ATTR_MAP(CTCLoss) = {
  {"preprocess_collapse_repeated", ATTR_DESC(preprocess_collapse_repeated, AnyTraits<bool>())},
  {"ctc_merge_repeated", ATTR_DESC(ctc_merge_repeated, AnyTraits<bool>())},
  {"ignore_longer_outputs_than_inputs", ATTR_DESC(ignore_longer_outputs_than_inputs, AnyTraits<bool>())}};
OUTPUT_MAP(CTCLoss) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(CTCLoss, kNameCTCLoss, ADPT_DESC(CTCLoss))

// CTCLossV2
INPUT_MAP(CTCLossV2) = {{1, INPUT_DESC(log_probs)},
                        {2, INPUT_DESC(targets)},
                        {3, INPUT_DESC(input_lengths)},
                        {4, INPUT_DESC(target_lengths)}};
ATTR_MAP(CTCLossV2) = {{"blank", ATTR_DESC(blank, AnyTraits<int64_t>())},
                       {"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())},
                       {"zero_infinity", ATTR_DESC(zero_infinity, AnyTraits<bool>())}};
OUTPUT_MAP(CTCLossV2) = {{0, OUTPUT_DESC(neg_log_likelihood)}, {1, OUTPUT_DESC(log_alpha)}};
REG_ADPT_DESC(CTCLossV2, prim::kPrimCTCLossV2->name(), ADPT_DESC(CTCLossV2))

// CTCLossV2Grad
INPUT_MAP(CTCLossV2Grad) = {{1, INPUT_DESC(grad_out)},       {2, INPUT_DESC(log_probs)},
                            {3, INPUT_DESC(targets)},        {4, INPUT_DESC(input_lengths)},
                            {5, INPUT_DESC(target_lengths)}, {6, INPUT_DESC(neg_log_likelihood)},
                            {7, INPUT_DESC(log_alpha)}};
ATTR_MAP(CTCLossV2Grad) = {{"blank", ATTR_DESC(blank, AnyTraits<int64_t>())},
                           {"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())},
                           {"zero_infinity", ATTR_DESC(zero_infinity, AnyTraits<bool>())}};
OUTPUT_MAP(CTCLossV2Grad) = {{0, OUTPUT_DESC(grad)}};
REG_ADPT_DESC(CTCLossV2Grad, prim::kPrimCTCLossV2Grad->name(), ADPT_DESC(CTCLossV2Grad))

// CTCGreedyDecoder
INPUT_MAP(CTCGreedyDecoder) = {{1, INPUT_DESC(inputs)}, {2, INPUT_DESC(sequence_length)}};
ATTR_MAP(CTCGreedyDecoder) = {{"merge_repeated", ATTR_DESC(merge_repeated, AnyTraits<bool>())}};
OUTPUT_MAP(CTCGreedyDecoder) = {{0, OUTPUT_DESC(decoded_indices)},
                                {1, OUTPUT_DESC(decoded_values)},
                                {2, OUTPUT_DESC(decoded_shape)},
                                {3, OUTPUT_DESC(log_probability)}};
REG_ADPT_DESC(CTCGreedyDecoder, kNameCTCGreedyDecoder, ADPT_DESC(CTCGreedyDecoder))
}  // namespace mindspore::transform
