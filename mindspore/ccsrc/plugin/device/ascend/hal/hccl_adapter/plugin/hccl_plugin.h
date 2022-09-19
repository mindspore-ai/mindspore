/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_PLUGIN_HCCL_PLUGIN_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_PLUGIN_HCCL_PLUGIN_H

#include <string>
#include <memory>
#include <map>
#include <functional>
#include "external/ge/ge_api_types.h"
#include "hccl/hccl.h"
#include "hccl/hcom.h"
#include "utils/dlopen_macro.h"

constexpr const char *kHcclOpsKernelInfoStore = "ops_kernel_info_hccl";

namespace ge {
class OpsKernelBuilder;
class OpsKernelInfoStore;
}  // namespace ge

extern "C" {
struct HcomOperation;
}  // extern C

using OptionsType = std::map<std::string, std::string>;
using OpsKernelBuilderMap = std::map<std::string, std::shared_ptr<ge::OpsKernelBuilder>>;
using HExecCallBack = std::function<void(HcclResult)>;

PLUGIN_METHOD(InitHcomGraphAdapter, ge::Status, const OptionsType &);
PLUGIN_METHOD(FinalizeHcomGraphAdapter, ge::Status);
PLUGIN_METHOD(GetHcclKernelInfoStore, void, std::shared_ptr<ge::OpsKernelInfoStore> *);
PLUGIN_METHOD(GetAllKernelBuilder, void, OpsKernelBuilderMap *);

ORIGIN_METHOD(HcclBroadcast, HcclResult, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclAllReduce, HcclResult, void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclReduceScatter, HcclResult, void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm,
              aclrtStream);
ORIGIN_METHOD(HcclAllGather, HcclResult, void *, void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclSend, HcclResult, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclRecv, HcclResult, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclAlltoAllV, HcclResult, const void *, const void *, const void *, HcclDataType, const void *,
              const void *, const void *, HcclDataType, HcclComm, aclrtStream);

ORIGIN_METHOD(HcclCommInitClusterInfo, HcclResult, const char *, uint32_t, HcclComm *);
ORIGIN_METHOD(HcclCommDestroy, HcclResult, HcclComm);
ORIGIN_METHOD(HcclGetRankId, HcclResult, void *, uint32_t *);
ORIGIN_METHOD(HcclGetRankSize, HcclResult, void *, uint32_t *);
ORIGIN_METHOD(HcomGetLocalRankId, HcclResult, const char *, uint32_t *);
ORIGIN_METHOD(HcomGetLocalRankSize, HcclResult, const char *, uint32_t *);
ORIGIN_METHOD(HcomGetWorldRankFromGroupRank, HcclResult, const char *, uint32_t, uint32_t *);
ORIGIN_METHOD(HcomGetGroupRankFromWorldRank, HcclResult, uint32_t, const char *, uint32_t *);

ORIGIN_METHOD(HcomCreateGroup, HcclResult, const char *, uint32_t, uint32_t *);
ORIGIN_METHOD(HcomDestroyGroup, HcclResult, const char *);
ORIGIN_METHOD(HcomGetRankId, HcclResult, const char *, uint32_t *);
ORIGIN_METHOD(HcomGetRankSize, HcclResult, const char *, uint32_t *);
ORIGIN_METHOD(HcomExecInitialize, HcclResult);
ORIGIN_METHOD(HcomExecFinalize, HcclResult);
ORIGIN_METHOD(HcomExecEnqueueOperation, HcclResult, ::HcomOperation, HExecCallBack);
ORIGIN_METHOD(HcomExecEnqueueAllToAllV, HcclResult, ::HcomAllToAllVParams, HExecCallBack);
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_PLUGIN_HCCL_PLUGIN_H
