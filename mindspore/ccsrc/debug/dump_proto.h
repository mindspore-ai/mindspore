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
#ifndef MINDSPORE_CCSRC_DEBUG_DUMP_PROTO_H_
#define MINDSPORE_CCSRC_DEBUG_DUMP_PROTO_H_

#include <string>

#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"
#include "debug/common.h"

namespace mindspore {
std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);

std::string GetOnnxProtoString(const FuncGraphPtr &func_graph);

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph);

mind_ir::ModelProto GetBinaryProto(const FuncGraphPtr &func_graph, bool save_tensor_data = false);

void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DUMP_PROTO_H_
