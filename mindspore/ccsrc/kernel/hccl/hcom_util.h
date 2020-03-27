/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_HCCL_HCOM_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_HCCL_HCOM_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "ir/dtype.h"
#include "hccl/base.h"

namespace mindspore {
using std::map;
using std::string;
using std::vector;

constexpr auto kAllGather = "AllGather";
constexpr auto kAllReduce = "AllReduce";
constexpr auto kBroadcast = "Broadcast";
constexpr auto kReduceScatter = "ReduceScatter";

/* Correspondence between data_type and hcom data type in Ascend */
static map<int64_t, hcclDataType_t> CONST_OP_HCOM_DATA_TYPE_MAP = {
  {TypeId::kNumberTypeFloat32, HCCL_DATA_TYPE_FLOAT},
  {TypeId::kNumberTypeFloat16, HCCL_DATA_TYPE_HALF},
  {TypeId::kNumberTypeInt8, HCCL_DATA_TYPE_INT8},
  {TypeId::kNumberTypeInt32, HCCL_DATA_TYPE_INT},
};

/* Correspondence between data_type and occupied byte size in hcom */
static map<hcclDataType_t, uint32_t> CONST_OP_HCOM_DATA_TYPE_SIZE_MAP = {
  {HCCL_DATA_TYPE_FLOAT, sizeof(float)},
  {HCCL_DATA_TYPE_HALF, sizeof(float) / 2},
  {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
  {HCCL_DATA_TYPE_INT, sizeof(int32_t)},
};

class HcomUtil {
 public:
  static bool GetKernelInputShape(const AnfNodePtr &anf_node, vector<vector<size_t>> *hccl_kernel_shape_list);
  static bool GetKernelOutputShape(const AnfNodePtr &anf_node, vector<vector<size_t>> *hccl_kernel_shape_list);
  static bool GetHcomDataType(const AnfNodePtr &anf_node, vector<hcclDataType_t> *data_type_list);
  static bool GetHcclOpSize(const hcclDataType_t &data_type, const vector<size_t> &shape, size_t *size);
  static bool GetHcomTypeSize(const hcclDataType_t &data_type, uint32_t *size);
  static bool GetHcomCount(const AnfNodePtr &anf_node, const vector<hcclDataType_t> &data_type_list,
                           const vector<vector<size_t>> &shape_list, uint64_t *total_count);
  static bool GetHcomOperationType(const AnfNodePtr &anf_node, hcclRedOp_t *op_type);
  static bool GetHcomRootId(const AnfNodePtr &anf_node, uint32_t *root_id);
};
}  // namespace mindspore

#endif
