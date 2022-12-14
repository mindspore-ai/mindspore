/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_CXXAPI_MODEL_AOE_AUTO_TUNE_PROCESS_H
#define MINDSPORE_CCSRC_CXXAPI_MODEL_AOE_AUTO_TUNE_PROCESS_H

#include <memory>
#include "include/transform/graph_ir/types.h"
#include "include/api/status.h"
#include "cxx_api/model/acl/acl_model_options.h"

namespace mindspore {
class AutoTuneProcess {
 public:
  static Status AoeOfflineTurningGraph(const std::weak_ptr<AclModelOptions> &options,
                                       const transform::DfGraphPtr &graph);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXXAPI_MODEL_AOE_AUTO_TUNE_PROCESS_H
