
/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/inference_weight_preprocess_utils.h"
#include <string>
#include <limits>
#include <memory>
#include <algorithm>
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace opt {
namespace {
template <typename SRC_T, typename DST_T>
void ConvertDataType(void *dst_data, void *ori_data, int64_t len, bool need_rank_offset, uint32_t global_rank_id) {
  auto rank_offset = need_rank_offset ? global_rank_id * len : 0;
  SRC_T *ori_data_t = reinterpret_cast<SRC_T *>(ori_data) + rank_offset;
  DST_T *dst_data_t = reinterpret_cast<DST_T *>(dst_data);
  for (int i = 0; i < len; i++) {
    dst_data_t[i] = static_cast<DST_T>(ori_data_t[i]);
  }
}

std::shared_ptr<ValueNode> CreateValueNode(const tensor::TensorPtr &assist_tensor, const TensorTypePtr &tensor_type) {
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  assist_tensor->set_device_info(device_info);
  MS_EXCEPTION_IF_NULL(assist_tensor);

  auto assist_const = std::make_shared<ValueNode>(assist_tensor);
  auto assist_abstract = assist_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({common::AnfAlgo::GetOutputInferDataType(assist_const, 0)});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), assist_const.get());
  return assist_const;
}

float int32_to_float(std::int32_t int_value) {
  union {
    std::int32_t i;
    float f;
  } converter;
  converter.i = int_value;
  return converter.f;
}

}  // namespace

std::shared_ptr<ValueNode> ConvertWeightsToNewType(const AnfNodePtr &weight_node) {
  auto w_param = GetParamFromLoad(weight_node->cast<CNodePtr>(), true);
  MS_EXCEPTION_IF_NULL(w_param);
  auto origin_shape = w_param->shape();
  auto shape = common::AnfAlgo::GetOutputInferShape(weight_node, kIndex0);
  if (shape.size() != 1 || origin_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "shape.size():" << shape.size() << " origin_shape.size():" << origin_shape.size()
                      << " not all == 1.";
  }
  bool need_rank_offset = false;
  if (origin_shape[0] != shape[0]) {
    need_rank_offset = true;
  }
  auto ori_data = w_param->data_c();
  auto w_type_id = static_cast<TypeId>(w_param->data_type_c());
  auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
  tensor::TensorPtr assist_tensor;
  TensorTypePtr tensor_type;
  if (w_type_id == kNumberTypeInt8) {
    assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape);
    tensor_type = std::make_shared<TensorType>(kInt32);
    ConvertDataType<int8_t, int32_t>(assist_tensor->data_c(), ori_data, shape[0], need_rank_offset, global_rank_id);
  } else if (w_type_id == kNumberTypeFloat16) {
    assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, shape);
    tensor_type = std::make_shared<TensorType>(kFloat32);
    ConvertDataType<float16, float>(assist_tensor->data_c(), ori_data, shape[0], need_rank_offset, global_rank_id);
  } else {
    MS_LOG(EXCEPTION) << "type_id " << TypeIdToString(w_type_id) << " is unexpected, only support int8 or fp16.";
  }

  return CreateValueNode(assist_tensor, tensor_type);
}

tensor::TensorPtr GetParamFromLoad(const CNodePtr &load, const bool unused) {
  if (IsPrimitiveCNode(load, prim::kPrimLoad)) {
    auto anf_node = common::AnfAlgo::GetInputNode(load, kIndex0);
    MS_EXCEPTION_IF_NULL(anf_node);
    if (anf_node->isa<Parameter>()) {
      auto para = anf_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(para);
      if (para->has_default()) {
        auto value = para->default_param();
        MS_EXCEPTION_IF_NULL(value);
        auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
        MS_EXCEPTION_IF_NULL(tensor);
        if (unused) {
          auto param_info = para->param_info();
          param_info->set_ignore_device_addr(true);
        }
        return tensor;
      }
    }
  }
  return nullptr;
}

bool CheckFusionValid(const CNodePtr &matmul, int64_t *k, const int trans_a_pos, const int trans_b_pos,
                      const std::vector<TypeId> &valid_dtypes) {
  auto inputs = matmul->inputs();
  auto trans_a_node = GetValueNode(inputs[trans_a_pos]);
  auto trans_b_node = GetValueNode(inputs[trans_b_pos]);
  MS_EXCEPTION_IF_NULL(trans_a_node);
  MS_EXCEPTION_IF_NULL(trans_b_node);
  bool trans_a = GetValue<bool>(trans_a_node);
  bool trans_b = GetValue<bool>(trans_b_node);
  if (trans_a != false) {
    return false;
  }
  if (trans_b != true) {
    return false;
  }
  auto weight_node = inputs[kIndex2]->cast<CNodePtr>();
  auto w_param = GetParamFromLoad(weight_node, false);
  if (!w_param) {
    return false;
  }
  auto w_type_id = static_cast<TypeId>(w_param->data_type_c());
  if (std::find(valid_dtypes.begin(), valid_dtypes.end(), w_type_id) == valid_dtypes.end()) {
    return false;
  }
  std::vector<int64_t> origin_shape = w_param->shape();
  auto parallel_shape = common::AnfAlgo::GetOutputInferShape(weight_node, kIndex0);
  // when param is not parallel tiled, it is not safe to use and concat, skip this pass
  if (parallel_shape.size() != origin_shape.size()) {
    return false;
  }
  for (int i = 0; i < static_cast<int>(parallel_shape.size()); i++) {
    if (parallel_shape[i] != origin_shape[i]) {
      return false;
    }
  }
  const int shape_num_two = 2;
  if (origin_shape.size() != shape_num_two) {
    return false;
  }
  if (*k == -1) {
    *k = origin_shape[1];
  } else if (*k != origin_shape[1]) {
    return false;
  }
  return true;
}

template <typename T>
void ConcatWeightsToNewTensor(void *data_ptr, const std::vector<void *> &data_c_list, const int64_t &k_len,
                              const std::vector<int64_t> &n_len_list, const bool &need_rank_offset,
                              const uint32_t &global_rank_id) {
  const auto data_size = sizeof(T);
  int64_t offset = 0;
  for (int idx = 0; idx < static_cast<int>(data_c_list.size()); idx++) {
    auto count = k_len * n_len_list[idx];
    auto rank_offset = need_rank_offset ? global_rank_id * count : 0;
    auto byte_size = count * data_size;
    memcpy_s(reinterpret_cast<T *>(data_ptr) + offset, byte_size, reinterpret_cast<T *>(data_c_list[idx]) + rank_offset,
             byte_size);
    offset += count;
  }
}

std::shared_ptr<ValueNode> CreateWeightTensor(TypeId type_id, const std::vector<int64_t> &weight_shape,
                                              const std::vector<void *> &data_c_list,
                                              const std::vector<int64_t> &n_len_list, const int64_t &k_len,
                                              const std::shared_ptr<Type> &w_dtype, const bool &need_rank_offset,
                                              const uint32_t &global_rank_id) {
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(type_id, weight_shape);
  auto data_ptr = assist_tensor->data_c();
  if (type_id == TypeId::kNumberTypeBFloat16) {
    ConcatWeightsToNewTensor<bfloat16>(data_ptr, data_c_list, k_len, n_len_list, need_rank_offset, global_rank_id);
  } else if (type_id == TypeId::kNumberTypeFloat16) {
    ConcatWeightsToNewTensor<float16>(data_ptr, data_c_list, k_len, n_len_list, need_rank_offset, global_rank_id);
  } else if (type_id == TypeId::kNumberTypeInt8) {
    ConcatWeightsToNewTensor<int8_t>(data_ptr, data_c_list, k_len, n_len_list, need_rank_offset, global_rank_id);
  }

  TensorTypePtr tensor_type = std::make_shared<TensorType>(w_dtype);
  return CreateValueNode(assist_tensor, tensor_type);
}

void SortWeightNodeList(AnfNodePtrList *node_list) {
  std::sort(node_list->begin(), node_list->end(), [](const AnfNodePtr &a, const AnfNodePtr &b) {
    auto para_a =
      common::AnfAlgo::GetInputNode(a->cast<CNodePtr>()->inputs()[2]->cast<CNodePtr>(), kIndex0)->cast<ParameterPtr>();
    auto para_b =
      common::AnfAlgo::GetInputNode(b->cast<CNodePtr>()->inputs()[2]->cast<CNodePtr>(), kIndex0)->cast<ParameterPtr>();
    return para_a->name() < para_b->name();
  });
}

std::shared_ptr<ValueNode> ConvertFp16BiasToInt32(const AnfNodePtr &bias_node, const AnfNodePtr &scale_node,
                                                  const bool &with_allreduce) {
  auto bias_param = GetParamFromLoad(bias_node->cast<CNodePtr>(), true);
  MS_EXCEPTION_IF_NULL(bias_param);
  auto scale_param = GetParamFromLoad(scale_node->cast<CNodePtr>(), false);
  MS_EXCEPTION_IF_NULL(scale_param);
  auto origin_shape = bias_param->shape();
  auto shape = common::AnfAlgo::GetOutputInferShape(bias_node, kIndex0);
  if (shape.size() != 1 || origin_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "shape.size():" << shape.size() << " origin_shape.size():" << origin_shape.size()
                      << " not all == 1.";
  }
  bool need_rank_offset = false;
  if (origin_shape[0] != shape[0]) {
    need_rank_offset = true;
  }
  void *bias_data = bias_param->data_c();
  void *scale_data = scale_param->data_c();
  auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt32);
  auto len = shape[0];

  auto rank_offset = need_rank_offset ? global_rank_id * len : 0;
  /**
   * logic:
   * (1) scale[int64] -> scale[int32] -> scale[float32];
   * (2) bias[fp16] -> bias[fp32];
   * (3) res = div(bias, scale) [fp64] -> round[fp64]
   * (4) res[fp64] -> res[int64] -> clamp[int32]
   */
  const double int32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
  const double int32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
  void *dst_data = assist_tensor->data_c();
  float16 *bias_data_t = reinterpret_cast<float16 *>(bias_data) + rank_offset;
  int64_t *scale_data_t = reinterpret_cast<int64_t *>(scale_data) + rank_offset;
  int32_t *dst_data_t = reinterpret_cast<int32_t *>(dst_data);
  for (int i = 0; i < len; i++) {
    if (global_rank_id == 0 || (!with_allreduce)) {
      int32_t scale_int32 = static_cast<int32_t>(scale_data_t[i]);
      float scale_fp32 = int32_to_float(scale_int32);
      double bias_fp64 = static_cast<double>(bias_data_t[i]);
      double res_fp64 = std::clamp(round(bias_fp64 / scale_fp32), int32_min, int32_max);
      dst_data_t[i] = static_cast<int32_t>(res_fp64);
    } else {
      dst_data_t[i] = 0;
    }
  }
  return CreateValueNode(assist_tensor, tensor_type);
}

}  // namespace opt
}  // namespace mindspore
