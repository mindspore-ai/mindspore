/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_
#define DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class InputValidationPass
/// \brief This is a parse pass that validates input parameters of the IR tree.
class InputValidationPass : public IRNodePass {
  /// \brief Runs a validation pass to check input parameters
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_
