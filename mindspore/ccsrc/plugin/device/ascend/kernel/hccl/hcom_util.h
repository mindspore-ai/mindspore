/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "ir/dtype.h"
#include "hccl/base.h"
#include "include/common/utils/contract.h"
#include "hccl/hccl_types.h"
#include "utils/shape_utils.h"

namespace mindspore {
using std::map;
using std::string;
using std::vector;
/* Correspondence between data_type and hcom data type in Ascend */
static map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
  {TypeId::kNumberTypeFloat32, HCCL_DATA_TYPE_FP32},
  {TypeId::kNumberTypeFloat16, HCCL_DATA_TYPE_FP16},
  {TypeId::kNumberTypeInt8, HCCL_DATA_TYPE_INT8},
  {TypeId::kNumberTypeInt32, HCCL_DATA_TYPE_INT32},
};

/* Correspondence between data_type and occupied byte size in hcom */
static map<HcclDataType, uint32_t> kConstOpHcomDataTypeSizeMap = {
  {HCCL_DATA_TYPE_FP32, sizeof(float)},
  {HCCL_DATA_TYPE_FP16, sizeof(float) / 2},
  {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
  {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},
};

class HcomUtil {
 public:
  static bool GetKernelInputShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_kernel_intput_shape_list);
  static bool GetKernelOutputShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_kernel_shape_list);
  static bool GetKernelInputInferShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_input_infer_shape_list);
  static bool GetKernelOutputInferShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_output_infer_shape_list);
  static ::HcclDataType ConvertHcclType(TypeId type_id);
  static bool GetHcomDataType(const AnfNodePtr &anf_node, vector<HcclDataType> *data_type_list);
  static bool GetHcclOpSize(const HcclDataType &data_type, const ShapeVector &shape, size_t *size);
  static bool GetHcomTypeSize(const HcclDataType &data_type, uint32_t *size);
  static bool GetHcomCount(const AnfNodePtr &anf_node, const vector<HcclDataType> &data_type_list,
                           const vector<ShapeVector> &shape_list, uint64_t *total_count);
  static bool GetHcomOperationType(const AnfNodePtr &anf_node, HcclReduceOp *op_type);
  static bool GetHcomRootId(const AnfNodePtr &anf_node, uint32_t *root_id);
  static bool GetHcomSrcRank(const AnfNodePtr &anf_node, uint32_t *src_rank);
  static bool GetHcomDestRank(const AnfNodePtr &anf_node, uint32_t *dest_rank);
  static void GetHcomGroup(NotNull<const AnfNodePtr &> anf_node, NotNull<std::string *> group);
  static bool GetHcomReceiveType(const AnfNodePtr &anf_node, TypeId *receive_type);
};
}  // namespace mindspore

#endif
