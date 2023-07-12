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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_AOE_API_TUNING_PROCESS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_AOE_API_TUNING_PROCESS_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "include/transform/graph_ir/types.h"
#include "include/api/status.h"
#include "cxx_api/model/acl/acl_model_options.h"
#include "external/ge/ge_api.h"

namespace mindspore {
using ConfigInfos = std::map<std::string, std::map<std::string, std::string>>;
class AoeApiTuning {
 public:
  Status AoeTurningGraph(const std::shared_ptr<ge::Session> &session, const transform::DfGraphPtr &graph,
                         const std::vector<ge::Tensor> &inputs, const std::shared_ptr<Context> &context,
                         const ConfigInfos &config_infos);

 private:
  std::map<std::string, std::string> GetAoeGlobalOptions(const std::shared_ptr<Context> &context,
                                                         const ConfigInfos &config_infos);
  std::map<std::string, std::string> GetAoeTuningOptions(const std::shared_ptr<Context> &context,
                                                         const ConfigInfos &config_infos);
  std::vector<std::string> GetAoeJobType(const std::shared_ptr<Context> &context, const ConfigInfos &config_infos);

  Status ExecuteAoe(const std::shared_ptr<ge::Session> &session, const transform::DfGraphPtr &graph,
                    const std::vector<ge::Tensor> &inputs, const std::vector<std::string> &job_types,
                    const std::map<std::string, std::string> &global_options,
                    const std::map<std::string, std::string> &tuning_options);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_AOE_API_TUNING_PROCESS_H_
