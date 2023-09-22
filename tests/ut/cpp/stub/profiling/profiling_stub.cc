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
#include "prof_callback.h"
#include "acl/acl_prof.h"
#include "toolchain/prof_api.h"

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

/*
 * @ingroup libprofapi
 * @name  MsprofSysCycleTime
 * @brief get systime cycle time of CPU
 * @return system cycle time of CPU
 */
MSVP_PROF_API uint64_t MsprofSysCycleTime() { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofReportEvent
 * @brief report event timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] event: event of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportEvent(uint32_t agingFlag, const MsprofEvent *event) { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofRegTypeInfo
 * @brief reg mapping info of type id and type name
 * @param [in] level: level is the report struct's level
 * @param [in] typeId: type id is the report struct's type
 * @param [in] typeName: label of type id for presenting user
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName) { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofReportCompactInfo
 * @brief report profiling compact information
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of compact information
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofReportAdditionalInfo
 * @brief report profiling additional information
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of additional information
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofGetHashId
 * @brief return hash id of hash info
 * @param [in] hashInfo: information to be hashed
 * @param [in] length: the length of information to be hashed
 * @return hash id
 */
MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length) { return 0; }

/*
 * @ingroup libprofapi
 * @name  MsprofReportApi
 * @brief report api timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] api: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api) { return 0; }
