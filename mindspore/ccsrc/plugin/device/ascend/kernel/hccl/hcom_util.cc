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
#include <memory>
#include "kernel/common_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
bool HcomUtil::GetKernelInputShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_kernel_intput_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_kernel_intput_shape_list);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
    hccl_kernel_intput_shape_list->emplace_back(shape_i);
  }

  return true;
}

bool HcomUtil::GetKernelOutputShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_kernel_output_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_kernel_output_shape_list);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  for (size_t i = 0; i < output_num; ++i) {
    auto shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    (void)hccl_kernel_output_shape_list->emplace_back(shape_i);
  }

  return true;
}

bool HcomUtil::GetKernelInputInferShape(const AnfNodePtr &anf_node, vector<ShapeVector> *hccl_input_infer_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_input_infer_shape_list);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto shape_i = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, i);
    (void)hccl_input_infer_shape_list->emplace_back(shape_i);
  }

  return true;
}

bool HcomUtil::GetKernelOutputInferShape(const AnfNodePtr &anf_node,
                                         vector<ShapeVector> *hccl_output_infer_shape_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(hccl_output_infer_shape_list);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  for (size_t i = 0; i < output_num; ++i) {
    auto shape_i = common::AnfAlgo::GetOutputInferShape(anf_node, i);
    (void)hccl_output_infer_shape_list->emplace_back(shape_i);
  }

  return true;
}

::HcclDataType HcomUtil::ConvertHcclType(TypeId type_id) {
  auto iter = kConstOpHcomDataTypeMap.find(type_id);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    MS_LOG(EXCEPTION) << "HcomDataType can't support Current Ascend Data Type : " << type_id;
  }
  return iter->second;
}

bool HcomUtil::GetHcomDataType(const AnfNodePtr &anf_node, vector<HcclDataType> *data_type_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(data_type_list);
  size_t tensor_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kReceiveOpName) {
    tensor_num = AnfAlgo::GetOutputTensorNum(anf_node);
  }
  for (size_t i = 0; i < tensor_num; ++i) {
    TypeId type_ptr;
    if (op_name == kReceiveOpName) {
      type_ptr = AnfAlgo::GetOutputDeviceDataType(anf_node, i);
    } else {
      type_ptr = AnfAlgo::GetInputDeviceDataType(anf_node, i);
    }
    data_type_list->emplace_back(ConvertHcclType(type_ptr));
  }
  if (!data_type_list->empty()) {
    if (std::any_of(data_type_list->begin(), data_type_list->end(),
                    [&data_type_list](HcclDataType type) { return type != *(data_type_list->begin()); })) {
      MS_LOG(ERROR) << "hccl have different data type";
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

bool HcomUtil::GetHcomCount(const AnfNodePtr &anf_node, const vector<HcclDataType> &data_type_list,
                            const vector<ShapeVector> &shape_list, uint64_t *total_count) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(total_count);
  const uint32_t align_size = 512;
  const uint32_t filled_size = 32;
  uint64_t total_size = 0;
  uint64_t block_size;
  size_t input_size;
  uint32_t type_size = 4;
  size_t size = common::AnfAlgo::GetInputTensorNum(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetCNodeName(anf_node) == kReceiveOpName) {
    size = AnfAlgo::GetOutputTensorNum(anf_node);
  }
  for (size_t i = 0; i < size; ++i) {
    if (!GetHcomTypeSize(data_type_list[i], &type_size)) {
      return false;
    }

    if (!GetHcclOpSize(data_type_list[i], shape_list[i], &input_size)) {
      MS_LOG(ERROR) << "Get GetHcclOpSize failed";
      return false;
    }

    if (common::AnfAlgo::GetCNodeName(anf_node) == kReduceScatterOpName) {
      int64_t rank_size;
      auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
      MS_EXCEPTION_IF_NULL(primitive);
      if (primitive->GetAttr(kAttrRankSize) != nullptr) {
        rank_size = GetValue<int64_t>(primitive->GetAttr(kAttrRankSize));
      } else {
        MS_LOG(ERROR) << "Get rank size failed";
        return false;
      }
      size_t actual_input_size = input_size;
      if (common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
          common::AnfAlgo::GetNodeAttr<int64_t>(anf_node, kAttrFusion) != 0) {
        actual_input_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
      }
      block_size = static_cast<uint64_t>(actual_input_size / LongToSize(rank_size));
      total_size = total_size + block_size;
    } else {
      if (common::AnfAlgo::GetCNodeName(anf_node) == kAllGatherOpName) {
        if (common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
            common::AnfAlgo::GetNodeAttr<int64_t>(anf_node, kAttrFusion) != 0 &&
            common::AnfAlgo::GetInputTensorNum(anf_node) > 1) {
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

bool HcomUtil::GetHcomOperationType(const AnfNodePtr &anf_node, HcclReduceOp *op_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_type);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr(kAttrOp) == nullptr) {
    MS_LOG(ERROR) << "Get HCOM_ATTR_REDUCE_TYPE fail, not support!";
    return false;
  }
  auto hcom_op_type = GetValue<std::string>(primitive->GetAttr(kAttrOp));
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

bool HcomUtil::GetHcomRootId(const AnfNodePtr &anf_node, uint32_t *root_id) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(root_id);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr(kAttrRootRank) != nullptr) {
    *root_id = static_cast<uint32_t>(GetValue<int64_t>(primitive->GetAttr(kAttrRootRank)));
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_ROOT_INDEX fail, not support!";
    return false;
  }
  return true;
}

bool HcomUtil::GetHcomSrcRank(const AnfNodePtr &anf_node, uint32_t *src_rank) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(src_rank);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("src_rank") != nullptr) {
    *src_rank = static_cast<uint32_t>(GetValue<int64_t>(primitive->GetAttr("src_rank")));
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_SRC_RANK fail, not support!";
    return false;
  }
  return true;
}

bool HcomUtil::GetHcomDestRank(const AnfNodePtr &anf_node, uint32_t *dest_rank) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(dest_rank);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("dest_rank") != nullptr) {
    *dest_rank = static_cast<uint32_t>(GetValue<int64_t>(primitive->GetAttr("dest_rank")));
  } else {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_DEST_RANK fail, not support!";
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
