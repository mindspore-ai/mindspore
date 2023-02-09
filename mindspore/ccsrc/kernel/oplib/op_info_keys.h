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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_KEYS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_KEYS_H_
#include <set>
#include <string>
#include <map>

namespace mindspore::kernel {
// json common value
constexpr auto kDynamicShapeSupport = "dynamicShapeSupport";
constexpr auto kFlag = "flag";
constexpr auto kTrue = "true";
constexpr auto kOp = "op";
constexpr auto kPattern = "pattern";
constexpr auto kSlicePattern = "slicePattern";
constexpr auto kValue = "value";
constexpr auto kNeedCheckSupport = "needCheckSupport";
constexpr auto kRangeLimit = "rangeLimit";
constexpr auto kGgatKeyAttr = "sgatKeyAttr";
constexpr auto kOpFile = "opFile";
constexpr auto kOpInterface = "opInterface";
constexpr auto kBinfile = "binfile";
constexpr auto kKernel = "kernel";
constexpr auto kAsync = "async";
constexpr auto kCompute = "compute";
constexpr auto kCost = "cost";
constexpr auto kDynamicFormat = "dynamicFormat";
constexpr auto kPartial = "partial";
constexpr auto kPrecisionReduce = "precision_reduce";
constexpr auto kDynamincRankSupport = "dynamicRankSupport";
constexpr auto kDynamicCompileStatic = "dynamicCompileStatic";
constexpr auto kHeavyOp = "heavyOp";
constexpr auto kCubeOp = "cubeOp";
constexpr auto kNull = "Null";
constexpr auto kJitCompile = "jitCompile";
constexpr auto kSoftSync = "softsync";
constexpr auto kOpImplSwitch = "opImplSwitch";
constexpr auto kPreBuildPattern = "prebuildPattern";

// attr keys
constexpr auto kAttr = "attr";
constexpr auto kList = "list";
constexpr auto kParamType = "paramType";
constexpr auto kType = "type";
constexpr auto kDefaultValue = "defaultValue";
// input/output keys
constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kDtype = "dtype";
constexpr auto kFormat = "format";
constexpr auto kName = "name";
constexpr auto kReshapeType = "reshapeType";
constexpr auto kReshape_Type = "reshape_type";
constexpr auto kShape = "shape";
constexpr auto kNeedCompile = "needCompile";
constexpr auto kUnknownShapeFormat = "unknownshape_format";
constexpr auto kValueDepend = "valueDepend";
constexpr auto kShapesType = "shapesType";
// key values
constexpr auto kRequired = "required";
constexpr auto kOptional = "optional";
constexpr auto kDynamic = "dynamic";
constexpr auto kImplyType = "imply_type";
constexpr auto kOpName = "op_name";
constexpr auto kAsyncFlag = "async_flag";
constexpr auto kComputeCost = "compute_cost";
constexpr auto kPartialFlag = "partial_flag";
constexpr auto kOpPattern = "op_pattern";
constexpr auto kFormatAgnostic = "formatAgnostic";
constexpr auto kBroadcast = "broadcast";
constexpr auto kReduce = "reduce";
constexpr auto kDtypeFormat = "dtype_format";
constexpr auto kIputs = "inputs";
constexpr auto kOutputs = "outputs";

constexpr auto kIndex = "index";
constexpr auto kProcessor = "processor";
constexpr auto kIgnored = "ignored";

constexpr auto kProAICORE = "AiCore";
constexpr auto kProCUDA = "CUDA";
constexpr auto kProCPU = "CPU";
constexpr auto kCtxCPU = "CPU";
constexpr auto kCtxGPU = "GPU";
constexpr auto kCtxAscend = "Ascend";
constexpr auto kImplyAKGStr = "AKG";  // this type refer: CUDA & CPU with difference process
constexpr auto kImplyTBEStr = "TBE";
constexpr auto kImplyAICPUStr = "AiCPU";
constexpr auto kImplyCPUStr = "CPU";
constexpr auto kImplyCUDAStr = "CUDA";
constexpr auto kImplyGPUStr = "GPU";
const std::map<std::string, std::string> kProcessorMap = {
  {kCtxAscend, kImplyTBEStr}, {kCtxGPU, kImplyCUDAStr}, {kCtxCPU, kImplyCPUStr}};
enum OpImplyType { kImplyAKG = 0, kImplyTBE, kImplyAICPU, kImplyCPU, kImplyGPU, kImplyBISHENG };
enum OpPattern { kCommonPattern = 0, kFormatAgnosticPattern, kBroadcastPattern, kReducePattern, kDynamicFormatPattern };
static const std::map<std::string, OpPattern> kPatternMap = {
  {kFormatAgnostic, kFormatAgnosticPattern},
  {kBroadcast, kBroadcastPattern},
  {kReduce, kReducePattern},
};
const std::map<std::string, OpImplyType> kImplyTypeStrToEnumMap = {{kImplyTBEStr, kImplyTBE},
                                                                   {kImplyAKGStr, kImplyAKG},
                                                                   {kImplyCPUStr, kImplyCPU},
                                                                   {kImplyAICPUStr, kImplyAICPU},
                                                                   {kImplyGPUStr, kImplyGPU}};
const std::map<OpImplyType, std::string> kImplyTypeEnumToStrMap = {{kImplyTBE, kImplyTBEStr},
                                                                   {kImplyAKG, kImplyAKGStr},
                                                                   {kImplyCPU, kImplyCPUStr},
                                                                   {kImplyAICPU, kImplyAICPUStr},
                                                                   {kImplyGPU, kImplyGPUStr}};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OP_INFO_KEYS_H_
