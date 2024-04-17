/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_PROF_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_PROF_SYMBOL_H_
#include <string>
#include "acl/acl_prof.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {
ORIGIN_METHOD(aclprofCreateConfig, aclprofConfig *, uint32_t *, uint32_t, aclprofAicoreMetrics,
              const aclprofAicoreEvents *, uint64_t)
ORIGIN_METHOD(aclprofDestroyConfig, aclError, const aclprofConfig *)
ORIGIN_METHOD(aclprofFinalize, aclError)
ORIGIN_METHOD(aclprofInit, aclError, const char *, size_t)
ORIGIN_METHOD(aclprofStart, aclError, const aclprofConfig *)
ORIGIN_METHOD(aclprofStop, aclError, const aclprofConfig *)
ORIGIN_METHOD(aclprofCreateStepInfo, aclprofStepInfo *)
ORIGIN_METHOD(aclprofGetStepTimestamp, aclError, aclprofStepInfo *, aclprofStepTag, aclrtStream)
ORIGIN_METHOD(aclprofDestroyStepInfo, void, aclprofStepInfo *)

extern aclprofCreateConfigFunObj aclprofCreateConfig_;
extern aclprofDestroyConfigFunObj aclprofDestroyConfig_;
extern aclprofFinalizeFunObj aclprofFinalize_;
extern aclprofInitFunObj aclprofInit_;
extern aclprofStartFunObj aclprofStart_;
extern aclprofStopFunObj aclprofStop_;
extern aclprofCreateStepInfoFunObj aclprofCreateStepInfo_;
extern aclprofGetStepTimestampFunObj aclprofGetStepTimestamp_;
extern aclprofDestroyStepInfoFunObj aclprofDestroyStepInfo_;

void LoadProfApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_PROF_SYMBOL_H_
