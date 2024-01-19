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
#include "acl/acl_prof.h"

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

