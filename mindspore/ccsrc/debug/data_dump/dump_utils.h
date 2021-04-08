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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_

#include <map>
#include <string>

#include "backend/session/kernel_graph.h"
#include "runtime/device/device_address.h"

namespace mindspore {
static const size_t PARAMETER_OUTPUT_INDEX = 0;
static const size_t VALUE_NODE_OUTPUT_INDEX = 0;

std::string GenerateDumpPath(uint32_t *device_id = nullptr);

void GetFileKernelName(NotNull<std::string *> kernel_name);

void GetConstantId(const session::KernelGraph *graph, std::map<std::string, size_t> *const_map);

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> int_shapes, bool trans_flag = false);

void DumpMemToFile(const std::string &file_path, NotNull<const device::DeviceAddress *> addr,
                   const ShapeVector &int_shapes, const TypeId &type, bool trans_flag = false);
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_
