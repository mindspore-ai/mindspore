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
#ifndef MINDSPORE_CCSRC_UTILS_CALLBACKS_H_
#define MINDSPORE_CCSRC_UTILS_CALLBACKS_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "transform/types.h"
#include "transform/util.h"

namespace mindspore {
namespace callbacks {

using mindspore::tensor::TensorPtr;
// define the python callback module and function
extern const char PYTHON_MOD_CALLBACK_MODULE[];
extern const char PYTHON_FUN_PROCESS_CHECKPOINT[];
extern const char PYTHON_FUN_PROCESS_SUMMARY[];

extern const char kSummary[];
extern const char kCheckPoint[];
extern const std::string kPythonCheckpointModuleName;
extern const std::string kPythonCheckpointFuncName;
bool GetParameterShape(const FuncGraphPtr& anf_graph, const std::string& param_name,
                       const std::shared_ptr<std::vector<int>>& shape);
uint32_t CheckpointSaveCallback(uint32_t, const std::map<std::string, ge::Tensor>&);
uint32_t SummarySaveCallback(uint32_t, const std::map<std::string, ge::Tensor>&);
uint32_t SummarySaveCallback(uint32_t, const std::map<std::string, TensorPtr>&);

}  // namespace callbacks
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CALLBACKS_H_
