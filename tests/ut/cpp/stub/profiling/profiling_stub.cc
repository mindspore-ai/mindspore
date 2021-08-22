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
#include <string>
#include "prof_mgr_core.h"

namespace Msprof {
namespace Engine {

class EngineIntf;
/**
 * @name  : RegisterEngine
 * @berif : API of libmsprof, register an engine with a name
 * @param [in]: module: the name of plugin
                engine: the plugin
 * @return: PROFILING_SUCCESS 0 (success)
 *          PROFILING_FAILED -1 (failed)
 */
int RegisterEngine(const std::string& module, const EngineIntf* engine) { return 0; }

}  // namespace Engine
}  // namespace Msprof

/**
 * @name  : ProfMgrStartUP
 * @berif : start Profiling task
 * @param : ProfMgrCfg cfg : config of start_up profiling
 * @return: NO_NULL (success)
 *        NULL (failed)
 */
void* ProfMgrStartUp(const ProfMgrCfg* cfg) { return const_cast<void*>(reinterpret_cast<const void*>(cfg)); }

/**
 * @name  : ProfMgrStop
 * @berif : stop Profiling task
 * @param : void * handle return by ProfMgrStartUP
 * @return: PROFILING_SUCCESS 0 (success)
 *        PROFILING_FAILED -1 (failed)
 */
int ProfMgrStop(void* handle) { return 0; }

namespace Analysis::Dvvp::ProfilerSpecial {
uint32_t MsprofilerInit() { return 0; }
}
