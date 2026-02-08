/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/hccl/hcom_util.h"
#include <set>
#include <algorithm>
#include <memory>
#include <utility>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "primitive/ascend_op_name.h"
#include "primitive/framework_op_name.h"
#include "primitive/other_op_name.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "ir/dtype/type.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
::HcclDataType HcomUtil::ConvertHcclType(TypeId type_id) {
  auto iter = kConstOpHcomDataTypeMap.find(type_id);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    if (type_id == TypeId::kNumberTypeComplex64) {
      MS_LOG(INFO) << "HcomDataType Can't support Current Ascend Data Type : Complex64, Convert it to Float32";
      return HCCL_DATA_TYPE_FP32;
    }
    MS_LOG(EXCEPTION) << "HcomDataType can't support Current Ascend Data Type : " << TypeIdLabel(type_id);
  }
  return iter->second;
}

void HcomUtil::AdjustShapeByDataType(TypeId type_id, ShapeVector *shape) {
  if (type_id == TypeId::kNumberTypeComplex64) {
    // When the input type is Complex64, the type is converted to Float32 and the shape is increased
    (void)shape->emplace_back(kComplex64ConvertFloat32Num);
  }
}

bool HcomUtil::GetHcomDataType(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                               const std::vector<KernelTensor *> &outputs, std::vector<HcclDataType> *data_type_list) {
  MS_EXCEPTION_IF_NULL(data_type_list);

  data_type_list->clear();
  const std::vector<KernelTensor *> &tensors = HcomUtil::IsReceiveOp(kernel_name) ? outputs : inputs;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(*data_type_list), [](KernelTensor *tensor_ptr) {
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return ConvertHcclType(tensor_ptr->dtype_id());
  });

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

bool HcomUtil::GetHcomCount(const PrimitivePtr &primitive, const std::vector<HcclDataType> &data_type_list,
                            const std::vector<ShapeVector> &shape_list, const size_t input_tensor_num,
                            const std::optional<int64_t> rank_size_opt, uint64_t *total_count) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(total_count);

  const uint32_t align_size = 512;
  const uint32_t filled_size = 32;
  uint64_t total_size = 0;
  size_t input_size;
  uint32_t type_size = 4;
  size_t rank_size = 1;
  static const std::set<string> &ops_name = {kReduceScatterOpName,
                                             ops::kNameInnerCommReduceScatter,
                                             ops::kNameDistCommReduceScatterTensor,
                                             ops::kNameDistCommScatterTensor,
                                             ops::kNameDistCommReduceScatter,
                                             ops::kNameDistCommScatter};
  bool is_reduce_scatter = (ops_name.count(primitive->name()) > 0);
  if (rank_size_opt.has_value()) {
    rank_size = LongToSize(rank_size_opt.value());
  } else if (is_reduce_scatter) {
    int64_t tmp_rank_size = 0;
    if (!HcomUtil::GetHcomAttr<int64_t>(primitive, kAttrRankSize, &tmp_rank_size)) {
      MS_LOG(ERROR) << "Get kAttrRankSize fail in " << primitive->name();
      return false;
    }
    rank_size = LongToSize(tmp_rank_size);
  }

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

    if (input_tensor_num > 1) {
      // communication operator with dynamic input should have continuous memory.
      MS_LOG(INFO) << "Communication operator " << primitive->name() << " has dynamic input.";
      input_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
    }

    if (is_reduce_scatter) {
      input_size /= rank_size;
    }
    bool all_dynamic = std::all_of(shape_list[i].begin(), shape_list[i].end(), [](int64_t x) { return x == -1; });
    if (!all_dynamic && (type_size == 0 || input_size % type_size != 0)) {
      MS_LOG(ERROR) << "primitive=" << primitive->name() << ", Input_size[" << input_size << "],Type_size[" << type_size
                    << "] != 0, fail!"
                    << " shape_list[i]=" << shape_list[i];
      return false;
    }
    total_size += input_size / type_size;
  }
  *total_count = total_size;
  return true;
}

std::pair<uint64_t, ::HcclDataType> HcomUtil::GetHcclCountAndTypeFromTensor(
  const PrimitivePtr &primitive, const tensor::TensorPtr &tensor, const std::optional<int64_t> rank_size_opt) {
  auto type_id = tensor->data_type();
  auto shape = tensor->shape();

  auto hccl_type = ConvertHcclType(type_id);
  AdjustShapeByDataType(type_id, &shape);

  uint64_t hccl_count = 0;
  constexpr size_t input_tensor_size = 1;
  if (!GetHcomCount(primitive, {hccl_type}, {shape}, input_tensor_size, rank_size_opt, &hccl_count)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }
  return std::make_pair(hccl_count, hccl_type);
}

bool HcomUtil::GetHcomOperationType(const PrimitivePtr &primitive, HcclReduceOp *op_type,
                                    device::CollectiveOpReduceType *collective_reduce_type) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(op_type);

  std::string hcom_op_type;
  if (!GetHcomAttr<std::string>(primitive, kAttrOp, &hcom_op_type)) {
    return false;
  }

  auto hcom_op_iter = kConstOpHcomReduceOpTypeMap.find(hcom_op_type);
  auto coll_op_iter = kConstOpCollectiveOpReduceTypeMap.find(hcom_op_type);
  if (hcom_op_iter == kConstOpHcomReduceOpTypeMap.end() || coll_op_iter == kConstOpCollectiveOpReduceTypeMap.end()) {
    MS_LOG(ERROR) << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << hcom_op_type << "] not support!";
    return false;
  }
  *op_type = hcom_op_iter->second;
  *collective_reduce_type = coll_op_iter->second;
  return true;
}

device::CollectiveOpReduceType HcomUtil::GetCollectiveOpReduceType(const std::string &reduce_op) {
  auto iter = kConstOpCollectiveOpReduceTypeMap.find(reduce_op);
  if (iter == kConstOpCollectiveOpReduceTypeMap.end()) {
    MS_LOG(EXCEPTION) << "HcomUtil::Get CollectiveOpReduceType fail, [" << reduce_op << "] not support!";
  }
  return iter->second;
}

HcclReduceOp HcomUtil::GetHcomReduceOpType(const std::string &reduce_op) {
  auto iter = kConstOpHcomReduceOpTypeMap.find(reduce_op);
  if (iter == kConstOpHcomReduceOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << reduce_op << "] not support!";
  }
  return iter->second;
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
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "Get Hcom Group Attr of Op:" << anf_node->fullname_with_scope()
                                          << " failed." << trace::DumpSourceLines(anf_node);
  }
}
}  // namespace mindspore
