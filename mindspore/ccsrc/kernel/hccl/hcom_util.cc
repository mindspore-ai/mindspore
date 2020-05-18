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

#include "kernel/hccl/hcom_util.h"

#include <memory>

#include "kernel/common_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"

namespace mindspore {
bool HcomUtil::GetKernelInputShape(const AnfNodePtr &anf_node, vector<vector<size_t>> *hccl_kernel_intput_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_kernel_intput_shape_list);
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(anf_node); ++i) {
    std::vector<size_t> shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
    hccl_kernel_intput_shape_list->emplace_back(shape_i);
  }

  return true;
}

bool HcomUtil::GetKernelOutputShape(const AnfNodePtr &anf_node, vector<vector<size_t>> *hccl_kernel_output_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_kernel_output_shape_list);
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(anf_node); ++i) {
    std::vector<size_t> shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    hccl_kernel_output_shape_list->emplace_back(shape_i);
  }

  return true;
}

bool HcomUtil::GetHcomDataType(const AnfNodePtr &anf_node, vector<hcclDataType_t> *data_type_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(data_type_list);
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(anf_node); ++i) {
    auto type_ptr = AnfAlgo::GetPrevNodeOutputDeviceDataType(anf_node, i);
    auto iter = CONST_OP_HCOM_DATA_TYPE_MAP.find(type_ptr);
    if (iter == CONST_OP_HCOM_DATA_TYPE_MAP.end()) {
      MS_LOG(EXCEPTION) << "HcomDataType cann't support Current Ascend Data Type : " << type_ptr;
    }
    data_type_list->emplace_back(iter->second);
  }
  auto type_base = *(std::begin(*data_type_list));
  if (std::any_of(data_type_list->begin(), data_type_list->end(),
                  [&type_base](hcclDataType_t type) { return type != type_base; })) {
    MS_LOG(ERROR) << "hccl have different data type";
    return false;
  }
  return true;
}

bool HcomUtil::GetHcclOpSize(const hcclDataType_t &data_type, const vector<size_t> &shape, size_t *size) {
  int tmp_size = 1;
  uint32_t type_size = 4;
  for (size_t i = 0; i < shape.size(); i++) {
    IntMulWithOverflowCheck(tmp_size, SizeToInt(shape[i]), &tmp_size);
  }

  if (!GetHcomTypeSize(data_type, &type_size)) {
    return false;
  }

  IntMulWithOverflowCheck(tmp_size, UintToInt(type_size), &tmp_size);
  *size = IntToSize(tmp_size);

  MS_LOG(INFO) << "size[" << *size << "]";
  return true;
}

bool HcomUtil::GetHcomTypeSize(const hcclDataType_t &data_type, uint32_t *size) {
  auto iter = CONST_OP_HCOM_DATA_TYPE_SIZE_MAP.find(data_type);
  if (iter == CONST_OP_HCOM_DATA_TYPE_SIZE_MAP.end()) {
    MS_LOG(ERROR) << "HcomUtil::HcomDataTypeSize, No DataTypeSize!";
    return false;
  }
  *size = iter->second;
  return true;
}

bool HcomUtil::GetHcomCount(const AnfNodePtr &anf_node, const vector<hcclDataType_t> &data_type_list,
                            const vector<vector<size_t>> &shape_list, uint64_t *total_count) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(total_count);
  const uint32_t align_size = 512;
  const uint32_t filled_size = 32;
  uint64_t total_size = 0;
  uint64_t block_size;
  size_t input_size;
  uint32_t type_size = 4;

  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(anf_node); ++i) {
    if (!GetHcomTypeSize(data_type_list[i], &type_size)) {
      return false;
    }

    if (!GetHcclOpSize(data_type_list[i], shape_list[i], &input_size)) {
      MS_LOG(ERROR) << "Get GetHcclOpSize failed";
      return false;
    }

    if (AnfAlgo::GetCNodeName(anf_node) == kReduceScatterOpName) {
      int32_t rank_size;
      auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
      MS_EXCEPTION_IF_NULL(primitive);
      if (primitive->GetAttr("rank_size") != nullptr) {
        rank_size = GetValue<int32_t>(primitive->GetAttr("rank_size"));
      } else {
        MS_LOG(ERROR) << "Get rank size failed";
        return false;
      }
      block_size = input_size / IntToSize(rank_size);
      total_size = total_size + block_size;
    } else {
      if (AnfAlgo::GetCNodeName(anf_node) == kAllGatherOpName) {
        block_size = input_size;
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

bool HcomUtil::GetHcomOperationType(const AnfNodePtr &anf_node, hcclRedOp_t *op_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_type);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("op") == nullptr) {
    MS_LOG(ERROR) << "Get HCOM_ATTR_REDUCE_TYPE fail, not support!";
    return false;
  }
  auto hcom_op_type_get = GetValue<const char *>(primitive->GetAttr("op"));
  string hcom_op_type(hcom_op_type_get);
  if (hcom_op_type == "min") {
    *op_type = HCCL_REP_OP_MIN;
  } else if (hcom_op_type == "max") {
    *op_type = HCCL_REP_OP_MAX;
  } else if (hcom_op_type == "prod") {
    *op_type = HCCL_REP_OP_PROD;
  } else if (hcom_op_type == "sum") {
    *op_type = HCCL_REP_OP_SUM;
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << hcom_op_type << "] not support!";
    return false;
  }
  return true;
}

bool HcomUtil::GetHcomRootId(const AnfNodePtr &anf_node, uint32_t *root_id) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(root_id);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("root_rank") != nullptr) {
    *root_id = (uint32_t)GetValue<int>(primitive->GetAttr("root_rank"));
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_ROOT_INDEX fail, not support!";
    return false;
  }
  return true;
}

void HcomUtil::GetHcomGroup(NotNull<const AnfNodePtr &> anf_node, NotNull<std::string *> group) {
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto attr = primitive->GetAttr("group");
  if (attr != nullptr) {
    *group = GetValue<std::string>(attr);
  } else {
    MS_LOG(EXCEPTION) << "Get Hcom Group Attr of Op:" << anf_node->fullname_with_scope() << " failed";
  }
}
}  // namespace mindspore
