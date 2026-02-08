/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/utils/callback.h"
#include "tools/debug_func.h"
#include "tools/silent_detect/checksum/checksum_mgr.h"
#include "tools/silent_detect/silent_detect_config_parser.h"
#include "tools/tensor_dump/tensordump_utils.h"
#include "tools/summary/summary.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/e2e_dump.h"
#include "tools/data_dump/dump_utils.h"
#include "tools/data_dump/debugger/proto_exporter.h"
#include "tools/data_dump/debugger/debugger.h"
namespace mindspore {
namespace tools {

using checksum::NeedEnableCheckSum;
using datadump::MbufTensorDumpCallback;
using silentdetect::IsSilentDetectEnable;

REGISTER_COMMON_CALLBACK(DebugOnStepBegin);
REGISTER_COMMON_CALLBACK(DebugPostLaunch);
REGISTER_COMMON_CALLBACK(DebugOnStepEnd);
REGISTER_COMMON_CALLBACK(DebugFinalize);
REGISTER_COMMON_CALLBACK(MbufTensorDumpCallback);
REGISTER_COMMON_CALLBACK(NeedEnableCheckSum);
REGISTER_COMMON_CALLBACK(IsSilentDetectEnable);
REGISTER_COMMON_CALLBACK(RecurseSetSummaryNodesForAllGraphs);
REGISTER_COMMON_CALLBACK(SummaryTensor);
REGISTER_COMMON_CALLBACK(RegisterSummaryCallBackFunc);
REGISTER_COMMON_CALLBACK(DumpJsonParserParse);
REGISTER_COMMON_CALLBACK(AsyncDumpEnabled);
REGISTER_COMMON_CALLBACK(E2eDumpEnabled);
REGISTER_COMMON_CALLBACK(DumpJsonParserPath);
REGISTER_COMMON_CALLBACK(GenerateDumpPath);
REGISTER_COMMON_CALLBACK(CopyDumpJsonToDir);
REGISTER_COMMON_CALLBACK(CopyMSCfgJsonToDir);
REGISTER_COMMON_CALLBACK(DumpJsonParserFinalize);
REGISTER_COMMON_CALLBACK(UpdateNeedDumpKernels);
REGISTER_COMMON_CALLBACK(DumpJsonParserDumpToFile);
REGISTER_COMMON_CALLBACK(InputNeedDump);
#ifdef ENABLE_DEBUGGER
REGISTER_COMMON_CALLBACK(DumpIRProtoWithSrcInfoDebugWholeStack);
REGISTER_COMMON_CALLBACK(DebuggerReset);
REGISTER_COMMON_CALLBACK(DebuggerInit);
REGISTER_COMMON_CALLBACK(DumpInGraphCompiler);
REGISTER_COMMON_CALLBACK(DebuggerBackendEnabled);
REGISTER_COMMON_CALLBACK(DebuggerLoadGraphs);
#endif
}  // namespace tools
}  // namespace mindspore
