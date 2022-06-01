/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_PARSER_UTILS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_PARSER_UTILS_H_

#include <set>
#include <string>
#include "mindapi/ir/common.h"
#include "mindapi/ir/anf.h"
#include "include/api/format.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/base/logging.h"
#include "include/errorcode.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/coded_stream.h"
using mindspore::lite::STATUS;
namespace mindspore {
namespace lite {
void GetAllFuncGraph(const api::FuncGraphPtr &func_graph, std::set<api::FuncGraphPtr> *all_func_graphs);
int PostAdjust(const std::set<api::FuncGraphPtr> &all_func_graphs);
int UnifyConvWeightFormat(const api::FuncGraphPtr &graph, const api::CNodePtr &cnode, mindspore::Format src_format,
                          mindspore::Format dst_format, std::set<api::AnfNodePtr> *has_visited);
bool ReadProtoFromCodedInputStream(google::protobuf::io::CodedInputStream *coded_stream,
                                   google::protobuf::Message *proto);
int ReadProtoFromText(const std::string &file, google::protobuf::Message *message);
int ReadProtoFromBinaryFile(const std::string &file, google::protobuf::Message *message);
STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_PARSER_UTILS_H_
