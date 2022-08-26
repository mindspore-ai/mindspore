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
#ifndef MINDSPORE_CCSRC_KERNEL_OPLIB_OP_INFO_CONSTANTS_H_
#define MINDSPORE_CCSRC_KERNEL_OPLIB_OP_INFO_CONSTANTS_H_

namespace mindspore::kernel::refactor {
// op info common
// dynamicShapeSupport
constexpr char kDynamicShapeSupport[] = "dynamicShapeSupport";
constexpr char kFlag[] = "flag";
constexpr char kTrue[] = "true";
constexpr char kFalse[] = "false";
// precision_reduce
constexpr char kPrecisionReduce[] = "precision_reduce";
// opFile
constexpr char kOpFile[] = "opFile";
constexpr char kValue[] = "value";
// opInterface
constexpr char kOpInterface[] = "opInterface";

// attr key
constexpr char kAttr[] = "attr";
constexpr char kList[] = "list";
// attr item key
constexpr char kDefaultValue[] = "defaultValue";
constexpr char kParamType[] = "paramType";
constexpr char kType[] = "type";
constexpr char kValue[] = "value";
// attr item value
constexpr char kInt[] = "int";
constexpr char kFloat[] = "float";
constexpr char kBool[] = "bool";
constexpr char kStr[] = "str";
constexpr char kListInt[] = "listInt";
constexpr char kListFloat[] = "listFloat";
constexpr char kListBool[] = "listBool";
constexpr char kListListInt[] = "listListInt";
constexpr char kRequired[] = "required";

// input/output
constexpr char kInputSuffix[] = "input";
constexpr char kDType[] = "dtype";
constexpr char kName[] = "name";
constexpr char kUnknownShapeFormat[] = "unknownshape_format";
constexpr char kFormat[] = "format";
}  // namespace mindspore::kernel::refactor
#endif  // MINDSPORE_CCSRC_KERNEL_OPLIB_OP_INFO_CONSTANTS_H_
