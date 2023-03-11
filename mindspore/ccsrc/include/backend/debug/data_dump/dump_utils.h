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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "include/backend/kernel_graph.h"
#include "include/backend/device_address.h"

using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;

namespace mindspore {
constexpr size_t kParameterOutputIndex = 0;
constexpr size_t kValueNodeOutputIndex = 0;

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Generate dir path to dump data. It will be in these formats:
 * 1) tensor/statistic: /dump_path/rank_{rank_id}/{net_name}/{graph_id}/{iter_num}.
 * 2) constant data: /dump_path/rank_{rank_id}/{net_name}/{graph_id}/constants/.
 */
std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id = 0, bool is_cst = false);

void GetFileKernelName(NotNull<std::string *> kernel_name);

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Get the actual tensor shape for dumping based on trans_flag option in configuration json file.
 */
void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> const int_shapes,
                     bool trans_flag = false);

const DeviceTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                       NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type);

/*
 * Feature group: Dump.
 * Target device group: Ascend, CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump the data in memory into file path.
 */
void DumpMemToFile(const std::string &file_path, const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                   const TypeId &type, bool trans_flag = false);

bool IsFolder(const std::string &file_path);

bool IsEmptyFolder(const std::string &dir_path);

void RemoveEmptyDir(const std::string &dir_path);

std::vector<std::string> Split(const std::string &input, const std::string &pattern);

void SaveOverflowOperator(const std::string &iterator, const std::string &dump_rank_path);

void DeleteNoOverflowFile(uint32_t rank_id, uint32_t graph_id);

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU, CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Remove scope from operator name. The default separator is "--".
 */
BACKEND_EXPORT std::string GetOpNameWithoutScope(const std::string &fullname_with_scope,
                                                 const std::string &separator = "--");

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU, CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump string content into file path. Current purpose is to save operator overflow information in json
 * file in ascend a+m dump mode.
 */
BACKEND_EXPORT void DumpToFile(const std::string &file_name, const std::string &dump_str);
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_UTILS_H_
