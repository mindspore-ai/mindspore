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

#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include <algorithm>
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "ops/ascend_op_name.h"
#include "ops/framework_op_name.h"
#include "ops/other_op_name.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
::HcclDataType HcomUtil::ConvertHcclType(TypeId type_id) {
  auto iter = kConstOpHcomDataTypeMap.find(type_id);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    MS_LOG(EXCEPTION) << "HcomDataType can't support Current Ascend Data Type : " << type_id;
  }
  return iter->second;
}

bool HcomUtil::GetHcomDataType(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                               const std::vector<KernelTensor *> &outputs, vector<HcclDataType> *data_type_list) {
  MS_EXCEPTION_IF_NULL(data_type_list);

  data_type_list->clear();
  const std::vector<KernelTensor *> &tensors = HcomUtil::IsReceiveOp(kernel_name) ? outputs : inputs;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(*data_type_list),
                 [](KernelTensor *tensor_ptr) { return ConvertHcclType(tensor_ptr->dtype_id()); });

  if (!data_type_list->empty()) {
    if (std::any_of(data_type_list->begin(), data_type_list->end(),
                    [&data_type_list](HcclDataType type) { return type != *(data_type_list->begin()); })) {
      MS_LOG(ERROR) << "hccl kernel " << kernel_name << " have different data type";
      return false;
    }
  }
  return true;
}

bool HcomUtil::GetHcclOpSize(const HcclDataType &data_type, const ShapeVector &shape, size_t *size) {
  MS_EXCEPTION_IF_NULL(size);
  int64_t tmp_size = 1;
  uint32_t type_size = 4;
  for (size_t i = 0; i < shape.size(); i++) {
    tmp_size = LongMulWithOverflowCheck(tmp_size, shape[i]);
  }

  if (!GetHcomTypeSize(data_type, &type_size)) {
    return false;
  }

  *size = SizetMulWithOverflowCheck(LongToSizeClipNeg(tmp_size), type_size);

  MS_LOG(DEBUG) << "size[" << *size << "]";
  return true;
}

bool HcomUtil::GetHcomTypeSize(const HcclDataType &data_type, uint32_t *size) {
  MS_EXCEPTION_IF_NULL(size);
  auto iter = kConstOpHcomDataTypeSizeMap.find(data_type);
  if (iter == kConstOpHcomDataTypeSizeMap.end()) {
    MS_LOG(ERROR) << "HcomUtil::HcomDataTypeSize, No DataTypeSize!";
    return false;
  }
  *size = iter->second;
  return true;
}

bool HcomUtil::GetHcomCount(const PrimitivePtr &primitive, const vector<HcclDataType> &data_type_list,
                            const vector<ShapeVector> &shape_list, const size_t input_tensor_num,
                            uint64_t *total_count) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(total_count);

  const uint32_t align_size = 512;
  const uint32_t filled_size = 32;
  uint64_t total_size = 0;
  uint64_t block_size;
  size_t input_size;
  uint32_t type_size = 4;

  MS_EXCEPTION_IF_CHECK_FAIL(data_type_list.size() == shape_list.size(),
                             "Size of data_type_list must be equal to size of shape_list");

  for (size_t i = 0; i < data_type_list.size(); ++i) {
    if (!GetHcomTypeSize(data_type_list[i], &type_size)) {
      return false;
    }

    if (!GetHcclOpSize(data_type_list[i], shape_list[i], &input_size)) {
      MS_LOG(ERROR) << "Get GetHcclOpSize failed";
      return false;
    }

    if (primitive->name() == kReduceScatterOpName) {
      int64_t rank_size;
      if (!HcomUtil::GetHcomAttr<int64_t>(primitive, kAttrRankSize, &rank_size)) {
        return false;
      }
      size_t actual_input_size = input_size;
      if (primitive->HasAttr(kAttrFusion) && GetValue<int64_t>(primitive->GetAttr(kAttrFusion)) != 0) {
        actual_input_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
      }
      block_size = static_cast<uint64_t>(actual_input_size / LongToSize(rank_size));
      total_size = total_size + block_size;
    } else {
      if (primitive->name() == kAllGatherOpName) {
        if (primitive->HasAttr(kAttrFusion) && GetValue<int64_t>(primitive->GetAttr(kAttrFusion)) != 0 &&
            input_tensor_num > 1) {
          block_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
        } else {
          block_size = input_size;
        }
      } else {
        block_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
      }
      total_size = total_size + block_size;
    }
  }

  if (type_size == 0 || total_size % type_size != 0) {
    MS_LOG(ERROR) << "Total_size[" << total_size << "],Type_size[" << type_size << "] != 0, fail!";
    return false;
  }
  *total_count = total_size / type_size;
  return true;
}

bool HcomUtil::GetHcomOperationType(const PrimitivePtr &primitive, HcclReduceOp *op_type) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(op_type);

  std::string hcom_op_type;
  if (!GetHcomAttr<std::string>(primitive, kAttrOp, &hcom_op_type)) {
    return false;
  }
  if (hcom_op_type == "min") {
    *op_type = HCCL_REDUCE_MIN;
  } else if (hcom_op_type == "max") {
    *op_type = HCCL_REDUCE_MAX;
  } else if (hcom_op_type == "prod") {
    *op_type = HCCL_REDUCE_PROD;
  } else if (hcom_op_type == "sum") {
    *op_type = HCCL_REDUCE_SUM;
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << hcom_op_type << "] not support!";
    return false;
  }
  return true;
}

bool HcomUtil::GetHcomReceiveType(const AnfNodePtr &anf_node, TypeId *receive_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(receive_type);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("dtype") != nullptr) {
    *receive_type = GetValue<NumberPtr>(primitive->GetAttr("dtype"))->type_id();
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_SRTAG_INDEX fail, not support!";
    return false;
  }
  return true;
}

void HcomUtil::GetHcomGroup(NotNull<const AnfNodePtr &> anf_node, NotNull<std::string *> group) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto attr = primitive->GetAttr(kAttrGroup);
  if (attr != nullptr) {
    *group = GetValue<std::string>(attr);
  } else {
    MS_LOG(EXCEPTION) << "Get Hcom Group Attr of Op:" << anf_node->fullname_with_scope() << " failed."
                      << trace::DumpSourceLines(anf_node);
  }
}
}  // namespace mindspore
