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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_

#include <memory>
#include <list>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class GetterPass
/// \brief This is a tree pass that will for now only clear the callback in MapOp to prevent hang
class GetterPass : public IRNodePass {
 public:
  /// \brief Default Constructor
  GetterPass() = default;

  /// \brief Default Destructor
  ~GetterPass() = default;

  Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
