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

#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "device_address/device_address.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/utils/contract.h"
#include "tools/visible.h"

using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;

namespace mindspore {
constexpr size_t kParameterOutputIndex = 0;
constexpr size_t kValueNodeOutputIndex = 0;

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: Convert int4 data_type into int8 data_type. The int4_data is 2 int4 data stored in 1 int8 data. After
 * split, the int8_data is 1 int4 data stored int 1 int8 data.
 */
TOOLS_EXPORT bool SplitInt8ToInt4x2(const void *int4_data, size_t in_data_len, void *int8_data, size_t out_data_len);

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

const kernel::KernelTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                               NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type);

/*
 * Feature group: Dump.
 * Target device group: Ascend, CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump the data in memory into file path.
 */
void DumpMemToFile(const std::string &file_path, const kernel::KernelTensorPtr &kernel_tensor,
                   const ShapeVector &int_shapes, const TypeId &type, bool trans_flag = false);

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MSBackend
 * Description: Load the device data into host mem.
 */
bool LoadMemToHost(kernel::KernelTensor *const kernel_tensor, const std::string &tensor_name,
                   const std::string &host_fmt, const ShapeVector &host_shape, TypeId host_type, size_t slot,
                   bool keep_prev, uint32_t root_graph_id, bool force_update, bool trans_flag, bool async_copy = True);

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MSBackend
 * Description: Extracts elements from the original tensor via specified shape, storage offset and strides,
 * then constructs a new contiguous tensor in host memory (output is contiguous regardless of the original layout).
 */
mindspore::tensor::TensorPtr ExtractContiguousTensor(const tensor::TensorPtr &ori_tensor, const ShapeVector &host_shape,
                                                     TypeId host_type, size_t storage_offset,
                                                     std::vector<int64_t> host_strides);
/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU, CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump string content into file path. Current purpose is to save operator overflow information in json
 * file in ascend a+m dump mode.
 */
TOOLS_EXPORT void DumpToFile(const std::string &file_name, const std::string &dump_str);
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_UTILS_H_
