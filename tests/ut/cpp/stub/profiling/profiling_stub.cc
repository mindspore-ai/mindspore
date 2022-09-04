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
#include "prof_callback.h"
#include "acl/acl_prof.h"

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
int RegisterEngine(const std::string &module, const EngineIntf *engine) { return 0; }

}  // namespace Engine
}  // namespace Msprof

/**
 * @name  : ProfMgrStartUP
 * @berif : start Profiling task
 * @param : ProfMgrCfg cfg : config of start_up profiling
 * @return: NO_NULL (success)
 *        NULL (failed)
 */
void *ProfMgrStartUp(const ProfMgrCfg *cfg) { return const_cast<void *>(reinterpret_cast<const void *>(cfg)); }

/**
 * @name  : ProfMgrStop
 * @berif : stop Profiling task
 * @param : void * handle return by ProfMgrStartUP
 * @return: PROFILING_SUCCESS 0 (success)
 *        PROFILING_FAILED -1 (failed)
 */
int ProfMgrStop(void *handle) { return 0; }

namespace Analysis::Dvvp::ProfilerSpecial {
uint32_t MsprofilerInit() { return 0; }
}  // namespace Analysis::Dvvp::ProfilerSpecial

/*
 * @name  MsprofInit
 * @brief Profiling module init
 * @param [in] dataType: profiling type: ACL Env/ACL Json/GE Option
 * @param [in] data: profiling switch data
 * @param [in] dataLen: Length of data
 * @return 0:SUCCESS, >0:FAILED
 */
int32_t MsprofInit(uint32_t dataType, void *data, uint32_t dataLen) { return 0; }

/*
 * @name AscendCL
 * @brief Finishing Profiling
 * @param NULL
 * @return 0:SUCCESS, >0:FAILED
 */
int32_t MsprofFinalize() { return 0; }

ACL_FUNC_VISIBILITY aclError aclprofInit(const char *profilerResultPath, size_t length) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclprofStart(const aclprofConfig *profilerConfig) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclprofStop(const aclprofConfig *profilerConfig) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclprofFinalize() { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclprofConfig *aclprofCreateConfig(uint32_t *deviceIdList, uint32_t deviceNums,
                                                       aclprofAicoreMetrics aicoreMetrics,
                                                       const aclprofAicoreEvents *aicoreEvents,
                                                       uint64_t dataTypeConfig) {
  return nullptr;
}

ACL_FUNC_VISIBILITY aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig) { return ACL_SUCCESS; }

/**
 * @name  profRegisterCallback
 * @brief register callback to profiling
 * @param moduleId  [IN] module Id
 * @param handle    [IN] the pointer of callback
 */
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) { return 0; }

/*
 * @name profReportData
 * @brief start reporter/stop reporter/report date
 * @param moduleId  [IN] enum profReporterModuleId
 * @param type      [IN] enum profReporterCallbackType
 * @param data      [IN] data (nullptr on INTI/UNINIT)
 * @param len       [IN] data size (0 on INIT/UNINIT)
 * @return enum MsprofErrorCod
 */
MSVP_PROF_API int32_t MsprofReportData(uint32_t moduleId, uint32_t type, void *data, uint32_t len) { return 0; }
