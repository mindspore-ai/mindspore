/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/candidate_sampling_op_declare.h"

namespace mindspore::transform {
// ComputeAccidentalHits
INPUT_MAP(ComputeAccidentalHits) = {{1, INPUT_DESC(true_classes)}, {2, INPUT_DESC(sampled_candidates)}};
ATTR_MAP(ComputeAccidentalHits) = {{"num_true", ATTR_DESC(num_true, AnyTraits<int64_t>())}};
OUTPUT_MAP(ComputeAccidentalHits) = {{0, OUTPUT_DESC(indices)}, {1, OUTPUT_DESC(ids)}, {2, OUTPUT_DESC(weights)}};
REG_ADPT_DESC(ComputeAccidentalHits, kNameComputeAccidentalHits, ADPT_DESC(ComputeAccidentalHits))

// UniformCandidateSampler
INPUT_MAP(UniformCandidateSampler) = {{1, INPUT_DESC(true_classes)}};
ATTR_MAP(UniformCandidateSampler) = {{"num_true", ATTR_DESC(num_true, AnyTraits<int64_t>())},
                                     {"num_sampled", ATTR_DESC(num_sampled, AnyTraits<int64_t>())},
                                     {"unique", ATTR_DESC(unique, AnyTraits<bool>())},
                                     {"range_max", ATTR_DESC(range_max, AnyTraits<int64_t>())},
                                     {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                     {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(UniformCandidateSampler) = {{0, OUTPUT_DESC(sampled_candidates)},
                                       {1, OUTPUT_DESC(true_expected_count)},
                                       {2, OUTPUT_DESC(sampled_expected_count)}};
REG_ADPT_DESC(UniformCandidateSampler, kNameUniformCandidateSampler, ADPT_DESC(UniformCandidateSampler))

}  // namespace mindspore::transform
