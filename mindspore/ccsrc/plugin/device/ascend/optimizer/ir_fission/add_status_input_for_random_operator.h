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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADD_STATUS_INPUT_FOR_RANDOM_OPERATOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADD_STATUS_INPUT_FOR_RANDOM_OPERATOR_H_

#include <set>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "ops/nn_op_name.h"
#include "ops/random_op_name.h"

namespace mindspore::opt {
inline const std::set<std::string> kRandomNodeWhiteList = {kDropoutGenMaskOpName,
                                                           kMultinomialOpName,
                                                           kParameterizedTruncatedNormalOpName,
                                                           kRandomCategoricalOpName,
                                                           kRandomChoiceWithMaskOpName,
                                                           kRandomPoissonOpName,
                                                           kRandomShuffleOpName,
                                                           kStandardLaplaceOpName,
                                                           kStandardNormalOpName,
                                                           kTruncatedNormalOpName,
                                                           kUniformOpName,
                                                           kUniformIntOpName,
                                                           kUniformRealOpName,
                                                           kUniformCandidateSamplerOpName,
                                                           kLogUniformCandidateSamplerOpName};

class AddStatusInputForRandomOperator : public Pass {
 public:
  AddStatusInputForRandomOperator() : Pass("add_status_input_for_random_operator") {}
  ~AddStatusInputForRandomOperator() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADD_STATUS_INPUT_FOR_RANDOM_OPERATOR_H_
