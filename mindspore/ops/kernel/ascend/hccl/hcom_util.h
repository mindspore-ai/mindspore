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
#include <unordered_map>
#include <utility>
#include "hccl/base.h"
#include "include/utils/contract.h"
#include "hccl/hccl_types.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "utils/shape_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "primitive/framework_op_name.h"
#include "primitive/ascend_op_name.h"
#include "ir/tensor.h"

namespace mindspore {
using kernel::KernelTensor;
using std::map;
using std::string;
using std::vector;
constexpr int64_t kComplex64ConvertFloat32Num = 2;

/* Correspondence between data_type and hcom data type in Ascend */
static const map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
  {TypeId::kNumberTypeBool, HCCL_DATA_TYPE_INT8},
  {TypeId::kNumberTypeInt8, HCCL_DATA_TYPE_INT8},
  {TypeId::kNumberTypeInt16, HCCL_DATA_TYPE_INT16},
  {TypeId::kNumberTypeInt32, HCCL_DATA_TYPE_INT32},
  {TypeId::kNumberTypeFloat16, HCCL_DATA_TYPE_FP16},
  {TypeId::kNumberTypeFloat32, HCCL_DATA_TYPE_FP32},
  {TypeId::kNumberTypeInt64, HCCL_DATA_TYPE_INT64},
  {TypeId::kNumberTypeUInt64, HCCL_DATA_TYPE_UINT64},
  {TypeId::kNumberTypeUInt8, HCCL_DATA_TYPE_UINT8},
  {TypeId::kNumberTypeUInt16, HCCL_DATA_TYPE_UINT16},
  {TypeId::kNumberTypeUInt32, HCCL_DATA_TYPE_UINT32},
  {TypeId::kNumberTypeFloat64, HCCL_DATA_TYPE_FP64},
  {TypeId::kNumberTypeBFloat16, HCCL_DATA_TYPE_BFP16},
#ifdef EXPERIMENT_A5
  {TypeId::kNumberTypeHiFloat8, HCCL_DATA_TYPE_HIF8},
  {TypeId::kNumberTypeFloat8E5M2, HCCL_DATA_TYPE_FP8E5M2},
  {TypeId::kNumberTypeFloat8E4M3FN, HCCL_DATA_TYPE_FP8E4M3},
#endif
};

/* Correspondence between data_type and occupied byte size in hcom */
static const map<HcclDataType, uint32_t> kConstOpHcomDataTypeSizeMap = {
  {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},         {HCCL_DATA_TYPE_INT16, sizeof(int32_t) / 2},
  {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},       {HCCL_DATA_TYPE_FP16, sizeof(float) / 2},
  {HCCL_DATA_TYPE_FP32, sizeof(float)},          {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
  {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)},     {HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
  {HCCL_DATA_TYPE_UINT16, sizeof(uint32_t) / 2}, {HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)},
  {HCCL_DATA_TYPE_FP64, sizeof(double)},         {HCCL_DATA_TYPE_BFP16, sizeof(float) / 2},
#ifdef EXPERIMENT_A5
  {HCCL_DATA_TYPE_HIF8, sizeof(float) / 4},      {HCCL_DATA_TYPE_FP8E5M2, sizeof(float) / 4},
  {HCCL_DATA_TYPE_FP8E4M3, sizeof(float) / 4},
#endif
};

/* Correspondence between reduce str and enum in hcom  */
static const std::unordered_map<std::string, HcclReduceOp> kConstOpHcomReduceOpTypeMap = {
  {"min", HCCL_REDUCE_MIN},
  {"max", HCCL_REDUCE_MAX},
  {"prod", HCCL_REDUCE_PROD},
  {"sum", HCCL_REDUCE_SUM},
};

/* Correspondence between reduce str and enum in collective op  */
static const std::unordered_map<std::string, device::CollectiveOpReduceType> kConstOpCollectiveOpReduceTypeMap = {
  {"min", device::CollectiveOpReduceType::Reduce_Min},
  {"max", device::CollectiveOpReduceType::Reduce_Max},
  {"prod", device::CollectiveOpReduceType::Reduce_Prod},
  {"sum", device::CollectiveOpReduceType::Reduce_Sum},
};

class HcomUtil {
 public:
  static ::HcclDataType ConvertHcclType(TypeId type_id);
  static bool GetHcomDataType(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                              const std::vector<KernelTensor *> &outputs, std::vector<HcclDataType> *data_type_list);
  static bool GetHcclOpSize(const HcclDataType &data_type, const ShapeVector &shape, size_t *size);
  static bool GetHcomTypeSize(const HcclDataType &data_type, uint32_t *size);
  static bool GetHcomCount(const PrimitivePtr &primitive, const std::vector<HcclDataType> &data_type_list,
                           const std::vector<ShapeVector> &shape_list, const size_t input_tensor_num,
                           const std::optional<int64_t> rank_size_opt, uint64_t *total_count);

  static std::pair<uint64_t, ::HcclDataType> GetHcclCountAndTypeFromTensor(
    const PrimitivePtr &primitive, const tensor::TensorPtr &tensor,
    const std::optional<int64_t> rank_size_opt = std::nullopt);
  static device::CollectiveOpReduceType GetCollectiveOpReduceType(const std::string &reduce_op);
  static HcclReduceOp GetHcomReduceOpType(const std::string &reduce_op);
  static bool GetHcomOperationType(const PrimitivePtr &primitive, HcclReduceOp *op_type,
                                   device::CollectiveOpReduceType *collective_reduce_type);
  static void GetHcomGroup(NotNull<const AnfNodePtr &> anf_node, NotNull<std::string *> group);
  static bool GetHcomReceiveType(const AnfNodePtr &anf_node, TypeId *receive_type);
  static void AdjustShapeByDataType(TypeId type_id, ShapeVector *shape);

  static inline bool IsReceiveOp(const std::string &kernel_name) { return kernel_name == mindspore::kReceiveOpName; }

  template <typename T>
  static inline bool GetHcomAttr(const PrimitivePtr &prim, const std::string &attr_name, T *value_ptr) {
    MS_EXCEPTION_IF_NULL(prim);
    MS_EXCEPTION_IF_NULL(value_ptr);

    ValuePtr attr_value = prim->GetAttr(attr_name);
    if (attr_value == nullptr) {
      MS_LOG(DEBUG) << "Get attribute '" << attr_name << "' of kernel " << prim->name() << " failed.";
      return false;
    }

    *value_ptr = GetValue<T>(attr_value);
    return true;
  }

  template <typename DstType, typename SrcType>
  static inline bool GetHcomAttr(const PrimitivePtr &prim, const std::string &attr_name, DstType *value_ptr) {
    MS_EXCEPTION_IF_NULL(prim);
    MS_EXCEPTION_IF_NULL(value_ptr);

    SrcType value;
    if (!GetHcomAttr<SrcType>(prim, attr_name, &value)) {
      return false;
    }

    *value_ptr = static_cast<DstType>(value);
    return true;
  }
};
}  // namespace mindspore

#endif
