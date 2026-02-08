/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/rms_norm_quant_fusion.h"

#include <cstring>
#include <vector>
#include <string>
#include <utility>

#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "utils/ms_context.h"
#include "primitive/nn_ops.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"

namespace mindspore {
namespace opt {
template <typename T>
std::shared_ptr<ValueNode> CreateZeroTensor(const ShapeVector &gamma_shape, TypeId gamma_type) {
  tensor::TensorPtr assist_tensor = tensor::from_spec(gamma_type, gamma_shape, device::DeviceType::kCPU);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(TypeIdToType(gamma_type));
  T *dst_data_t = reinterpret_cast<T *>(assist_tensor->data_c());
  const auto data_size = sizeof(T);
  auto set_ret = memset_s(dst_data_t, gamma_shape[0] * data_size, 0, gamma_shape[0] * data_size);
  if (set_ret != EOK) {
    MS_LOG(EXCEPTION) << "Failed to set tensor to zeros.";
  }
  return CreateValueNode(assist_tensor, tensor_type);
}

template <typename T>
std::shared_ptr<ValueNode> CreateNewGammaTensor(const AnfNodePtr &gamma, const AnfNodePtr &scale) {
  auto gamma_param = GetParamFromLoad(gamma->cast<CNodePtr>(), true);
  MS_EXCEPTION_IF_NULL(gamma_param);
  auto scale_param = GetParamFromLoad(scale->cast<CNodePtr>(), true);
  MS_EXCEPTION_IF_NULL(scale_param);
  TypeId gamma_type = common::AnfAlgo::GetOutputInferDataType(gamma, 0);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(TypeIdToType(gamma_type));
  auto origin_shape = gamma_param->shape();
  auto shape = common::AnfAlgo::GetOutputInferShape(gamma, kIndex0);
  bool need_rank_offset = false;
  if (origin_shape[0] != shape[0]) {
    need_rank_offset = true;
  }
  auto len = shape[0];
  auto gamma_param_cpu = gamma_param->cpu();
  void *gamma_data = gamma_param_cpu->data_c();
  auto scale_param_cpu = scale_param->cpu();
  void *scale_data = scale_param_cpu->data_c();
  auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
  auto rank_offset = need_rank_offset ? global_rank_id * len : 0;
  tensor::TensorPtr assist_tensor = tensor::from_spec(gamma_type, shape, device::DeviceType::kCPU);
  void *dst_data = assist_tensor->data_c();

  T *gamma_data_t = reinterpret_cast<T *>(gamma_data) + rank_offset;
  T *scale_data_t = reinterpret_cast<T *>(scale_data) + rank_offset;
  T *dst_data_t = reinterpret_cast<T *>(dst_data);
  for (int i = 0; i < len; i++) {
    float gamma_fp32 = static_cast<float>(gamma_data_t[i]);
    float scale_fp32 = static_cast<float>(scale_data_t[i]);
    float new_gamma_fp32 = gamma_fp32 / scale_fp32;
    dst_data_t[i] = static_cast<T>(new_gamma_fp32);
  }
  return CreateValueNode(assist_tensor, tensor_type);
}

template <typename T>
std::shared_ptr<ValueNode> CreateScaleTensor(TypeId gamma_type) {
  ShapeVector shape = {1};
  tensor::TensorPtr assist_tensor = tensor::from_spec(gamma_type, shape, device::DeviceType::kCPU);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(TypeIdToType(gamma_type));
  T *dst_data_t = reinterpret_cast<T *>(assist_tensor->data_c());
  dst_data_t[0] = static_cast<T>(1.0);
  return CreateValueNode(assist_tensor, tensor_type);
}

std::shared_ptr<ValueNode> CreateOffsetTensor(const AnfNodePtr &offset) {
  auto offset_param = GetParamFromLoad(offset->cast<CNodePtr>(), true);
  MS_EXCEPTION_IF_NULL(offset_param);
  auto offset_param_cpu = offset_param->cpu();
  void *offset_data = offset_param_cpu->data_c();
  int8_t *offset_data_t = reinterpret_cast<int8_t *>(offset_data);

  ShapeVector shape = {1};
  tensor::TensorPtr assist_tensor = tensor::from_spec(kNumberTypeInt8, shape, device::DeviceType::kCPU);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(TypeIdToType(kNumberTypeInt8));
  int8_t *dst_data_t = reinterpret_cast<int8_t *>(assist_tensor->data_c());
  dst_data_t[0] = offset_data_t[0];
  return CreateValueNode(assist_tensor, tensor_type);
}

inline bool IsZero(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto value_ptr = utils::cast<ValueNodePtr>(n);

    if (value_ptr == nullptr) {
      return false;
    }

    auto idx_value = GetValue<int64_t>(value_ptr->value());
    if (idx_value == 0) {
      return true;
    }
  }

  return false;
}

static bool IsSupport(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &rms_norm) {
  auto x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm, 0);
  auto gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm, 1);
  auto scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  auto offset_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 2);

  if (x_dtype != kNumberTypeFloat16 && x_dtype != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "RmsNormQuant fused failed because of unsupported x_dtype: " << x_dtype;
    return false;
  }

  if (x_dtype != gamma_dtype || x_dtype != scale_dtype) {
    MS_LOG(INFO) << "RmsNormQuant fused failed because of  inconsistent dtype, x_dtype: " << x_dtype
                 << ", gamma_dtype: " << gamma_dtype << ", scale_dtype: " << scale_dtype;
    return false;
  }

  if (offset_dtype != kNumberTypeInt8) {
    MS_LOG(INFO) << "RmsNormQuant fused failed because of  unsupported offset_dtype: " << offset_dtype;
    return false;
  }

  auto gamma_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(rms_norm, kIndex1);
  if (gamma_shape.size() != 1) {
    MS_LOG(INFO) << "RmsNormQuant fused failed because the rank of gamma is not 1:" << gamma_shape;
    return false;
  }

  // if rstd is used, do not fuse
  auto rms_norm_users = GetRealNodeUsedList(graph, rms_norm);
  for (const auto &user : *rms_norm_users) {
    const auto &get_item_node = user.first;
    if (!IsPrimitiveCNode(get_item_node, prim::kPrimTupleGetItem)) {
      continue;
    }

    auto get_item_index = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(get_item_node), 1);
    auto get_item_index_value_ptr = get_item_index->cast<ValueNodePtr>();
    if (get_item_index_value_ptr == nullptr) {
      MS_LOG(INFO) << "RmsNormQuant fused failed because the index in TupleGetItem is not constant: "
                   << get_item_index->DebugString() << ", get_item_node: " << get_item_node->DebugString();
      return false;
    }

    constexpr auto kRstdIndexInRmsNormOut = 1;
    auto index_value = GetValue<int64_t>(get_item_index_value_ptr->value());
    if (index_value == kRstdIndexInRmsNormOut) {
      auto rstd_users = GetRealNodeUsedList(graph, get_item_index);
      if (rstd_users->size() != 0) {
        MS_LOG(INFO) << "RmsNormQuant fused failed because rstd is used by some node";
        return false;
      }

      break;
    }
  }

  auto scale_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  auto offset_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 2);

  if (scale_shape.size() > 1 || IsDynamicRank(scale_shape) || offset_shape.size() > 1 || IsDynamicRank(offset_shape)) {
    MS_LOG(INFO)
      << "RmsNormQuant fused failed because the rank of scale_shape and offset_shape must be 1, but got scale_shape: "
      << scale_shape << ", offset_shape: " << offset_shape;
    return false;
  }

  return true;
}

static const AnfNodePtr CreateRmsNormQuantNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &x1,
                                               const AnfNodePtr &gamma, const AnfNodePtr &beta, const AnfNodePtr &scale,
                                               const AnfNodePtr &offset, const AnfNodePtr &eps,
                                               const std::vector<AnfNodePtr> &orig_nodes) {
  auto prim = std::make_shared<Primitive>("RmsNormQuant");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, gamma, beta, scale, offset, eps};
  auto rms_norm_quant = NewCNode(inputs, graph, orig_nodes);
  MS_EXCEPTION_IF_NULL(rms_norm_quant);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  auto output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, rms_norm_quant.get());
  rms_norm_quant->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(rms_norm_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, rms_norm_quant.get());

  return rms_norm_quant;
}

std::vector<std::string> RmsNormQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimQuantV2->name()};
  return ret;
}

const BaseRef RmsNormQuantFusion::DefinePattern() const {
  auto index0 = std::make_shared<CondVar>(IsConstant);
  auto rms_norm = VectorRef({prim::kPrimRmsNorm, x1_, gamma_, eps_});

  auto tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, rms_norm, index0});

  auto sqrt_mode0 = std::make_shared<CondVar>(IsConstant);      // not used
  auto rounding_mode0 = std::make_shared<CondVar>(IsConstant);  // not used
  auto dst_type0 = std::make_shared<CondVar>(IsConstant);       // not used
  auto quant =
    VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale0_, offset0_, sqrt_mode0, rounding_mode0, dst_type0});
  return quant;
}

const AnfNodePtr RmsNormQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "Internal op is disabled.";
    return nullptr;
  }

  const std::string fusion_op_name = "RmsNormQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    MS_LOG(INFO) << "Internal RmsNormQuant is disabled.";
    return nullptr;
  }

  auto rms_norm_out0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out0), 0);
  MS_EXCEPTION_IF_NULL(rms_norm_node);

  if (!IsSupport(graph, node, rms_norm_node)) {
    MS_LOG(INFO) << "Can't fused to RmsNormQuant because of unsupported case.";
    return nullptr;
  }

  auto rms_norm_out0_users = GetRealNodeUsedList(graph, rms_norm_out0);
  if (rms_norm_out0_users->size() > 1) {
    MS_LOG(INFO) << "RmsNormQuant fused failed because the number of users of rms_norm_out0 is more than 1: "
                 << rms_norm_out0_users->size();
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale0_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset0_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  TypeId gamma_type = common::AnfAlgo::GetOutputInferDataType(gamma, 0);
  auto gamma_shape = common::AnfAlgo::GetOutputInferShape(gamma, kIndex0);
  TypeId scale_type = common::AnfAlgo::GetOutputInferDataType(scale, 0);
  auto scale_shape = common::AnfAlgo::GetOutputInferShape(scale, kIndex0);
  if (gamma_shape.size() != 1 || scale_shape.size() != 1) {
    MS_LOG(INFO) << "gamma_shape.size():" << gamma_shape.size() << " scale_shape.size():" << scale_shape.size()
                 << " != 1.";
    return nullptr;
  }

  if (gamma_type != scale_type || gamma_shape[0] != scale_shape[0]) {
    MS_LOG(INFO) << "gamma_type:" << TypeIdToString(gamma_type) << " scale_type:" << TypeIdToString(scale_type)
                 << "gamma_shape[0]:" << gamma_shape[0] << " scale_shape[0]:" << scale_shape[0] << " not equal.";
    return nullptr;
  }
  ValueNodePtr beta;
  ValueNodePtr new_gamma;
  ValueNodePtr new_scale;
  ValueNodePtr new_offset;
  if (gamma_type == kNumberTypeFloat16) {
    beta = CreateZeroTensor<float16>(gamma_shape, gamma_type);
    new_gamma = CreateNewGammaTensor<float16>(gamma, scale);
    new_scale = CreateScaleTensor<float16>(gamma_type);
  } else if (gamma_type == kNumberTypeBFloat16) {
    beta = CreateZeroTensor<bfloat16>(gamma_shape, gamma_type);
    new_gamma = CreateNewGammaTensor<bfloat16>(gamma, scale);
    new_scale = CreateScaleTensor<bfloat16>(gamma_type);
  } else {
    MS_LOG(INFO) << "gamma_type:" << TypeIdToString(gamma_type) << " != kNumberTypeFloat16 && != kNumberTypeBFloat16.";
    return nullptr;
  }
  if (!beta) {
    MS_LOG(INFO) << "beta is nullptr.";
    return nullptr;
  }
  new_offset = CreateOffsetTensor(offset);
  kernel_graph->AddValueNodeToGraph(beta);
  kernel_graph->AddValueNodeToGraph(new_gamma);
  kernel_graph->AddValueNodeToGraph(new_scale);
  kernel_graph->AddValueNodeToGraph(new_offset);

  auto rms_norm_quant =
    CreateRmsNormQuantNode(graph, node, x1, new_gamma, beta, new_scale, new_offset, eps, orig_nodes_);
  if (rms_norm_quant != nullptr) {
    MS_LOG(INFO) << "RmsNormQuant fused successfully.";
  } else {
    MS_LOG(INFO) << "RmsNormQuant fused failed.";
  }

  return rms_norm_quant;
}

std::vector<std::string> RmsNormAddQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name(), prim::kPrimQuantV2->name()};
  return ret;
}

const BaseRef RmsNormAddQuantFusion::DefinePattern() const {
  auto index0 = std::make_shared<CondVar>(IsConstant);
  auto rms_norm = VectorRef({prim::kPrimRmsNorm, x1_, gamma_, eps_});

  auto tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, rms_norm, index0});
  auto add = VectorRef({prim::kPrimAdd, tuple_get_item_0, beta0_});

  auto sqrt_mode0 = std::make_shared<CondVar>(IsConstant);      // not used
  auto rounding_mode0 = std::make_shared<CondVar>(IsConstant);  // not used
  auto dst_type0 = std::make_shared<CondVar>(IsConstant);       // not used
  auto quant = VectorRef({prim::kPrimQuantV2, add, scale0_, offset0_, sqrt_mode0, rounding_mode0, dst_type0});
  return quant;
}

static constexpr auto kRmsNormOut2OneAddQuant = 1;
static constexpr auto kRmsNormOut2TwoAddQuant = 2;
static constexpr auto kRmsNormOut2OneAddQuantAndOneShape = 3;
static constexpr auto kRmsNormOut2TwoAddQuantAndOneShape = 4;

// the num of Add is more than kAddNumTwo, or one of the users is not add-quant
static constexpr auto kUnsupportedTag = 0xffff;

void GetAddAndShapeNum(const FuncGraphPtr &graph,
                       const std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> &rms_norm_out0_users,
                       size_t *add_num, size_t *shape_num, AnfNodePtr *shape_node) {
  for (const auto &user : *rms_norm_out0_users) {
    const auto &user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimAdd)) {
      auto add_users = GetRealNodeUsedList(graph, user_node);
      if (add_users->size() != 1) {
        MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the user of Add is more than one: " << add_users->size();
        return;
      }

      if (!IsPrimitiveCNode(add_users->at(0).first, prim::kPrimQuantV2)) {
        MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the user of Add is not Quant: "
                     << add_users->at(0).first->fullname_with_scope();
        return;
      }
      ++(*add_num);
    } else if (IsPrimitiveCNode(user_node, prim::kPrimShape)) {
      ++(*shape_num);
      *shape_node = user_node;
    }
  }
}

inline size_t GetOpsCaseAfterRmsNorm(const FuncGraphPtr &graph, const AnfNodePtr &rms_norm_out0,
                                     AnfNodePtr *shape_node) {
  auto users = GetRealNodeUsedList(graph, rms_norm_out0);

  auto user_num = users->size();
  size_t add_num = 0;
  size_t shape_num = 0;

  GetAddAndShapeNum(graph, users, &add_num, &shape_num, shape_node);

  if (user_num == 1) {
    if (add_num != 1) {
      MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the user of RmsNorm is not Add-Quant";
      return kUnsupportedTag;
    }

    return kRmsNormOut2OneAddQuant;
  }

  if (user_num == 2) {
    if (shape_num == 1 && add_num == 1) {
      return kRmsNormOut2OneAddQuantAndOneShape;
    }

    if (add_num == user_num) {
      return kRmsNormOut2TwoAddQuant;
    }

    MS_LOG(INFO)
      << "RmsNormAddQuant fuse failed because the num of Add and shape in users of RmsNorm is invalid, add_num: "
      << add_num << ", shape_num: " << shape_num;
    return kUnsupportedTag;
  }

  if (user_num == 3) {
    if (shape_num == 1 && add_num == 2) {
      return kRmsNormOut2TwoAddQuantAndOneShape;
    }

    MS_LOG(INFO)
      << "RmsNormAddQuant fuse failed because the num of Add and shape in users of RmsNorm is invalid, add_num: "
      << add_num << ", shape_num: " << shape_num;
    return kUnsupportedTag;
  }

  return kUnsupportedTag;
}

const AnfNodePtr RmsNormAddQuantFusion::RmsNormQuantFuseWithOnePath(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                                    const EquivPtr &equiv,
                                                                    const AnfNodePtr &shape_node) const {
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[beta0_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale0_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset0_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  AnfNodePtr shape_input_node = nullptr;
  if (shape_node != nullptr) {
    shape_input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(shape_node), 0);
    if (shape_input_node == nullptr) {
      MS_LOG(INFO) << "RmsNormAddQuant fused failed because shape_input_node is nullptr";
      return nullptr;
    }
  }

  auto rms_norm_quant = CreateRmsNormQuantNode(graph, node, x1, gamma, beta, scale, offset, eps, orig_nodes_);

  if (shape_node != nullptr) {
    auto mng = graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    (void)mng->Replace(shape_input_node, rms_norm_quant);
  }

  return rms_norm_quant;
}

#define IsByteAlign(size, len) ((size & (len - 1)) == 0)

template <typename T>
inline bool DataNotEqual(void *data_c0, void *data_c1, size_t size) {
  auto ptr0 = reinterpret_cast<T *>(data_c0);
  auto ptr1 = reinterpret_cast<T *>(data_c1);

  auto elem = size / sizeof(T);
  for (size_t i = 0; i < elem; ++i) {
    if (ptr0[i] != ptr1[i]) {
      return true;
    }
  }

  return false;
}

inline bool ValueNotEqual(void *data_c0, void *data_c1, size_t size) {
  if (IsByteAlign(size, 8)) {
    return DataNotEqual<int64_t>(data_c0, data_c1, size);
  }

  if (IsByteAlign(size, 4)) {
    return DataNotEqual<int32_t>(data_c0, data_c1, size);
  }

  if (IsByteAlign(size, 2)) {
    return DataNotEqual<int16_t>(data_c0, data_c1, size);
  }

  if (IsByteAlign(size, 1)) {
    return DataNotEqual<int8_t>(data_c0, data_c1, size);
  }

  return true;
}

inline bool ParameterNotEqual(const std::string &name, const AnfNodePtr &load0, const AnfNodePtr &load1) {
  if (load0 == load1) {
    return false;
  }

  auto load_input0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(load0), 0);
  auto param_ptr0 = load_input0->cast<ParameterPtr>();
  auto load_input1 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(load1), 0);
  auto param_ptr1 = load_input1->cast<ParameterPtr>();

  if (param_ptr0 == nullptr || param_ptr1 == nullptr) {
    MS_LOG(INFO) << "One of the parameter is nullptr for " << name << ", param_ptr0: " << param_ptr0
                 << ", param_ptr1: " << param_ptr1;
    return true;
  }

  if (!param_ptr0->has_default() || !param_ptr1->has_default()) {
    MS_LOG(INFO) << "One of the parameter does not have default value for " << name
                 << ", param_ptr0: " << param_ptr0->has_default() << ", param_ptr1: " << param_ptr1->has_default();
    return true;
  }

  auto value_ptr0 = param_ptr0->default_param();
  auto value_ptr1 = param_ptr1->default_param();

  if (value_ptr0 == nullptr || value_ptr1 == nullptr) {
    MS_LOG(INFO) << "One of the value is nullptr for " << name << ", value_ptr0: " << value_ptr0
                 << ", value_ptr1: " << value_ptr1;
    return true;
  }

  auto tensor_ptr0 = value_ptr0->cast<tensor::TensorPtr>();
  auto tensor_ptr1 = value_ptr1->cast<tensor::TensorPtr>();

  if (tensor_ptr0 == nullptr || tensor_ptr1 == nullptr) {
    MS_LOG(INFO) << "One of the Tensor is nullptr for " << name << ", tensor_ptr0: " << tensor_ptr0
                 << ", tensor_ptr1: " << tensor_ptr1;
    return true;
  }

  auto tensor_ptr0_cpu = tensor_ptr0->cpu();
  auto data_c0 = tensor_ptr0_cpu->data_c();
  auto tensor_ptr1_cpu = tensor_ptr1->cpu();
  auto data_c1 = tensor_ptr1_cpu->data_c();

  if (data_c0 == nullptr || data_c1 == nullptr) {
    MS_LOG(INFO) << "One of the data_c is nullptr for " << name << ", data_c0: " << data_c0 << ", data_c1: " << data_c1;
    return true;
  }

  auto size0 = tensor_ptr0->Size();
  auto size1 = tensor_ptr0->Size();
  if (size0 != size1) {
    MS_LOG(INFO) << "The size is not equal, size0: " << size0 << ", size1: " << size1;
    return true;
  }

  return ValueNotEqual(data_c0, data_c1, size0);
}

const AnfNodePtr RmsNormAddQuantFusion::RmsNormQuantFuseWithTwoPath(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                                    const EquivPtr &equiv,
                                                                    const AnfNodePtr &rms_norm_out0,
                                                                    const AnfNodePtr &shape_node) const {
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta0_load = utils::cast<AnfNodePtr>((*equiv)[beta0_]);
  auto scale0 = utils::cast<AnfNodePtr>((*equiv)[scale0_]);
  auto offset0 = utils::cast<AnfNodePtr>((*equiv)[offset0_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  AnfNodePtr shape_input_node = nullptr;
  if (shape_node != nullptr) {
    shape_input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(shape_node), 0);
    if (shape_input_node == nullptr) {
      MS_LOG(INFO) << "RmsNormAddQuant fused failed because shape_input_node is nullptr";
      return nullptr;
    }
  }

  AnfNodePtr beta1_load = nullptr;
  auto rms_norm_out0_users = GetRealNodeUsedList(graph, rms_norm_out0);

  AnfNodePtr second_path_add_node = nullptr;
  AnfNodePtr add0_node = nullptr;
  auto beta0_load_users = GetRealNodeUsedList(graph, beta0_load);
  for (const auto &user : *beta0_load_users) {
    if (IsPrimitiveCNode(utils::cast<CNodePtr>(user.first), prim::kPrimAdd)) {
      add0_node = user.first;
      break;
    }
  }

  for (const auto &user : *rms_norm_out0_users) {
    if (user.first == shape_node) {
      continue;
    }

    const auto add_node = user.first;
    auto load = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(add_node), 1);
    if (!IsPrimitiveCNode(load, prim::kPrimLoad)) {
      MS_LOG(INFO)
        << "RmsNormAddQuant fuse failed because the input node is not load when add-quant number is 2, input: "
        << load->DebugString();
      return nullptr;
    }

    if (add0_node == add_node) {
      continue;
    }

    beta1_load = load;
    second_path_add_node = add_node;
    break;
  }

  if (ParameterNotEqual("beta", beta0_load, beta1_load)) {
    MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the value of beta is not equal.";
    return nullptr;
  }

  if (second_path_add_node == nullptr) {
    return nullptr;
  }

  auto add1_users = GetRealNodeUsedList(graph, second_path_add_node);
  const auto &quant_node1 = add1_users->at(0).first;
  static constexpr auto kScaleIdx = 1;
  static constexpr auto kOffsetIdx = 2;
  auto scale1 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(quant_node1), kScaleIdx);
  auto offset1 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(quant_node1), kOffsetIdx);

  if (ParameterNotEqual("scale", scale0, scale1)) {
    MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the value of scale is not equal.";
    return nullptr;
  }

  if (ParameterNotEqual("offset", offset0, offset1)) {
    MS_LOG(INFO) << "RmsNormAddQuant fuse failed because the value of offset is not equal.";
    return nullptr;
  }

  auto rms_norm_quant_node =
    CreateRmsNormQuantNode(graph, node, x1, gamma, beta0_load, scale0, offset0, eps, orig_nodes_);

  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  (void)mng->Replace(quant_node1, rms_norm_quant_node);

  if (shape_node != nullptr) {
    (void)mng->Replace(shape_input_node, rms_norm_quant_node);
  }
  return rms_norm_quant_node;
}

const AnfNodePtr RmsNormAddQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "Internal op is disabled.";
    return nullptr;
  }

  const std::string fusion_op_name = "RmsNormQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    MS_LOG(INFO) << "Internal RmsNormQuant is disabled.";
    return nullptr;
  }

  auto add_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_out0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(add_node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out0), 0);
  MS_EXCEPTION_IF_NULL(rms_norm_node);

  if (!IsSupport(graph, node, rms_norm_node)) {
    MS_LOG(INFO) << "Can't fused to RmsNormAddQuant because of unsupported case.";
    return nullptr;
  }

  AnfNodePtr shape_node = nullptr;
  AnfNodePtr out_node = nullptr;
  auto num_of_add_after_rmsnorm = GetOpsCaseAfterRmsNorm(graph, rms_norm_out0, &shape_node);
  if (num_of_add_after_rmsnorm == kRmsNormOut2OneAddQuant ||
      num_of_add_after_rmsnorm == kRmsNormOut2OneAddQuantAndOneShape) {
    out_node = RmsNormQuantFuseWithOnePath(graph, node, equiv, shape_node);
  } else if (num_of_add_after_rmsnorm == kRmsNormOut2TwoAddQuant ||
             num_of_add_after_rmsnorm == kRmsNormOut2TwoAddQuantAndOneShape) {
    out_node = RmsNormQuantFuseWithTwoPath(graph, node, equiv, rms_norm_out0, shape_node);
  }

  if (out_node != nullptr) {
    MS_LOG(INFO) << "RmsNormAddQuant fused successfully with RmsNorm out case: " << num_of_add_after_rmsnorm;
  }

  return out_node;
}
}  // namespace opt
}  // namespace mindspore
