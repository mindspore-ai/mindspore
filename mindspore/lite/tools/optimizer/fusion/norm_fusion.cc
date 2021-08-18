/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/norm_fusion.h"
#include <memory>
#include <algorithm>
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "mindspore/core/ops/instance_norm.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "src/ops/ops_utils.h"
#include "src/common/prim_util.h"
#include "tools/optimizer/graph/node_infershape.h"

namespace mindspore {
namespace opt {
namespace {
STATUS GetReduceAxes(const BaseRef &n, std::vector<int> *axes) {
  MS_ASSERT(node != nullptr);
  if (utils::isa<ParameterPtr>(n)) {
    auto axes_param = utils::cast<ParameterPtr>(n);
    if (!axes_param->has_default() || axes_param->default_param() == nullptr) {
      return lite::RET_NOT_SUPPORT;
    }
    auto axes_value = axes_param->default_param()->cast<tensor::TensorPtr>();
    if (axes_value == nullptr) {
      return lite::RET_ERROR;
    }
    axes->resize(axes_value->shape()[0]);
    if (memcpy_s(axes->data(), axes_value->Size(), axes_value->data_c(), axes_value->Size()) == EOK) {
      return lite::RET_OK;
    }
  }
  if (utils::isa<ValueNodePtr>(n)) {
    auto axes_value_node = utils::cast<ValueNodePtr>(n);
    auto axes_content = CastToInt(axes_value_node->value());
    if (memcpy_s(axes->data(), axes_content.size() * sizeof(int), axes_content.data(),
                 axes_content.size() * sizeof(int)) == EOK) {
      return lite::RET_OK;
    }
  }
  return lite::RET_ERROR;
}

bool IsReduceNode(const EquivPtr &equiv, const VarPtr &input_prim, const VarPtr &input_axes, std::vector<int> *axes) {
  MS_ASSERT(equiv != nullptr && input_prim != nullptr);
  MS_ASSERT(input_axes != nullptr && axes != nullptr);
  auto reduce_value = utils::cast<AnfNodePtr>((*equiv)[input_prim]);
  MS_ASSERT(reduce_value != nullptr);
  auto mean2_primitive = GetValueNode<std::shared_ptr<ops::ReduceFusion>>(reduce_value);
  if (mean2_primitive == nullptr || mean2_primitive->GetAttr(ops::kMode) == nullptr ||
      mean2_primitive->get_mode() != mindspore::Reduce_Mean) {
    return false;
  }
  if (GetReduceAxes((*equiv)[input_axes], axes) != lite::RET_OK) {
    return false;
  }
  return true;
}
}  // namespace

CNodePtr NormFusion::CreateNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                    const schema::PrimitiveType type, float epsilon, int begin_norm_axis,
                                    int begin_params_axis) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto norm_primitive = std::make_unique<schema::PrimitiveT>();
  norm_primitive->value.type = type;
  PrimitiveCPtr primitive = nullptr;
  if (type == schema::PrimitiveType_LayerNormFusion) {
    auto layer_norm_primitive = std::make_shared<ops::LayerNormFusion>();
    layer_norm_primitive->Init(begin_norm_axis, begin_params_axis, epsilon, true);
    primitive = layer_norm_primitive;
  } else if (type == schema::PrimitiveType_InstanceNorm) {
    auto instance_norm_primitive = std::make_shared<ops::InstanceNorm>();
    instance_norm_primitive->Init(epsilon);
    primitive = instance_norm_primitive;
  } else {
    return nullptr;
  }
  auto value_node = NewValueNode(primitive);
  std::vector<AnfNodePtr> new_node_inputs = {value_node};
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  new_node_inputs.push_back(input_node);
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_ASSERT(gamma_node != nullptr);
  new_node_inputs.push_back(gamma_node);
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_ASSERT(beta_node != nullptr);
  new_node_inputs.push_back(beta_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  return new_node;
}

bool NormFusion::GetNormTypeAndAxis(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode,
                                    const std::vector<int> &mean_axes, const std::vector<int> &params_shape,
                                    schema::PrimitiveType *type, int *begin_norm_axis, int *begin_params_axis) const {
  MS_ASSERT(input_node != nullptr);
  MS_ASSERT(type != nullptr);
  MS_ASSERT(begin_norm_axis != nullptr);
  MS_ASSERT(begin_params_axis != nullptr);
  auto abstract = input_cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(DEBUG) << "abstract of input is nullptr";
    return false;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract should be abstract tensor";
    return false;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(DEBUG) << "Shape of Abstract should be ShapePtr";
    return false;
  }
  auto shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  int shape_size = shape.size();
  if (shape.empty()) {
    auto shape_size_map = ShapeSizeInfer(func_graph);
    if (shape_size_map.find(input_cnode->fullname_with_scope()) != shape_size_map.end()) {
      shape_size = shape_size_map[input_cnode->fullname_with_scope()];
    }
  }

  for (size_t i = 1; i < mean_axes.size(); ++i) {
    if (mean_axes[i] != mean_axes[i - 1] + 1) {
      MS_LOG(DEBUG) << "mean axes is not continuous";
      return false;
    }
  }
  const int kInstanceNormShapeSize = 4;
  const int kInstanceNormMeanAxesSize = 2;
  const int kInstanceNormMeanAxes0 = 1;
  const int kInstanceNormMeanAxes1 = 2;
  if (shape_size == kInstanceNormShapeSize && mean_axes.size() == kInstanceNormMeanAxesSize &&
      mean_axes[0] == kInstanceNormMeanAxes0 && mean_axes[1] == kInstanceNormMeanAxes1) {
    if (params_shape.size() == 1 && params_shape.back() == shape.back()) {
      *type = schema::PrimitiveType_InstanceNorm;
      return true;
    }
  }
  if (mean_axes.back() >= 0 && mean_axes.back() + 1 != shape_size) {
    MS_LOG(DEBUG) << "mean node is not reduce to last axis.";
    return false;
  }

  // there is no need to check params_shape
  *begin_norm_axis = mean_axes.front();
  if (*begin_norm_axis >= 0) {
    *begin_params_axis = shape_size - static_cast<int>(params_shape.size());
    if (*begin_params_axis < 0) {
      MS_LOG(DEBUG) << "LayerNorm begin_params_axis illegal, not fuse";
      return false;
    }
  } else {
    *begin_params_axis = -static_cast<int>(params_shape.size());
  }

  *type = schema::PrimitiveType_LayerNormFusion;
  return true;
}

bool NormFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, schema::PrimitiveType *type,
                              float *epsilon, int *begin_norm_axis, int *begin_params_axis) const {
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(epsilon != nullptr);
  MS_ASSERT(type != nullptr);
  MS_ASSERT(begin_norm_axis != nullptr);
  MS_ASSERT(begin_params_axis != nullptr);
  // beta
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_ASSERT(beta_node != nullptr);
  if (CheckIfNodeIsParam(beta_node) != lite::RET_OK) {
    return false;
  }
  auto beta_param = beta_node->cast<ParameterPtr>()->default_param();
  auto beta_tensor = std::dynamic_pointer_cast<tensor::Tensor>(beta_param);
  std::vector<int> beta_shape;
  std::transform(beta_tensor->shape().begin(), beta_tensor->shape().end(), std::back_inserter(beta_shape),
                 [](int64_t val) { return static_cast<int>(val); });
  // gamma
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_ASSERT(gamma_node != nullptr);
  if (CheckIfNodeIsParam(gamma_node) != lite::RET_OK) {
    return false;
  }
  auto gamma_param = gamma_node->cast<ParameterPtr>()->default_param();
  auto gamma_tensor = std::dynamic_pointer_cast<tensor::Tensor>(gamma_param);
  std::vector<int> gamma_shape;
  std::transform(gamma_tensor->shape().begin(), gamma_tensor->shape().end(), std::back_inserter(gamma_shape),
                 [](int64_t val) { return static_cast<int>(val); });
  // epsilon
  auto epsilon_node = utils::cast<AnfNodePtr>((*equiv)[epsilon_]);
  MS_ASSERT(epsilon_node != nullptr);
  if (CheckIfNodeIsParam(epsilon_node) != lite::RET_OK) {
    return false;
  }
  auto epsilon_param = epsilon_node->cast<ParameterPtr>()->default_param();
  auto epsilon_tensor = std::dynamic_pointer_cast<tensor::Tensor>(epsilon_param);
  auto epsilon_data = reinterpret_cast<float *>(epsilon_tensor->data_c());
  auto epsilon_shape = epsilon_tensor->shape();
  // mean2
  std::vector<int> mean2_axes;
  if (!IsReduceNode(equiv, mean2_, mean2_axes_, &mean2_axes)) {
    return false;
  }
  // mean1
  std::vector<int> mean1_axes;
  if (!IsReduceNode(equiv, mean1_, mean1_axes_, &mean1_axes)) {
    return false;
  }
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  if (!utils::isa<CNodePtr>(input_node)) {
    return false;
  }
  auto input_cnode = input_node->cast<CNodePtr>();
  if (mean1_axes != mean2_axes) {
    return false;
  }
  if (gamma_shape != beta_shape) {
    return false;
  }
  if (epsilon_shape.empty() || (epsilon_shape.size() == 1 && epsilon_shape[0] == 1)) {
    *epsilon = epsilon_data[0];
  } else {
    return false;
  }

  return GetNormTypeAndAxis(func_graph, input_cnode, mean1_axes, gamma_shape, type, begin_norm_axis, begin_params_axis);
}

int CommonShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 0);
  return in_shape_size.at(0);
}

int ExpandDimsShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 0);
  return in_shape_size.at(0) + 1;
}

int StridedSliceShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 0);
  auto new_axis_mask = primitive.value.AsStridedSlice()->new_axis_mask;
  auto add_dims = 0;
  while (new_axis_mask != 0) {
    new_axis_mask = (new_axis_mask - 1) & new_axis_mask;
    add_dims++;
  }
  return in_shape_size.at(0) + add_dims;
}

int MatMulShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 1);
  return in_shape_size[0];
}

int ReShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 1);
  return in_shape_size[1];
}

int StackSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 1);
  return std::accumulate(in_shape_size.begin(), in_shape_size.end(), 0);
}

int SqueezeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 0);
  auto axis = primitive.value.AsSqueeze()->axis;
  if (axis.empty()) {
    return 0;
  }
  return in_shape_size.at(0) - axis.size();
}

int OneHotSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 0);
  return in_shape_size.at(0) + 1;
}

int FillShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 1);
  return in_shape_size.at(1);
}

int ShapeOpSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) { return 1; }

int BroadcastShapeSizeInfer(const std::vector<int> &in_shape_size, const schema::PrimitiveT &primitive) {
  MS_ASSERT(in_shape_size.size() > 1);
  int result = 0;
  for (auto shape_size : in_shape_size) {
    result = std::max(result, shape_size);
  }
  return result;
}

std::map<string, int> NormFusion::ShapeSizeInfer(const FuncGraphPtr &func_graph) const {
  MS_ASSERT(func_graph != nullptr);
  std::map<string, int> node_shape_size;
  std::map<string, std::vector<int>> node_shape;
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto origin_primc = GetValueNode<PrimitiveCPtr>(cnode->input(0));
    auto prim_t = lite::GetPrimitiveT(cnode->input(0));
    if (prim_t == nullptr || shape_size_infer_registry_.find(prim_t->value.type) == shape_size_infer_registry_.end()) {
      continue;
    }
    auto prim_type = prim_t->value.type;
    // specific op infer shape
    if (prim_type == schema::PrimitiveType_Shape) {
      tensor::TensorPtr tensor_info;
      auto ret = GetTensorInfoFromAbstract(&tensor_info, cnode, 1);
      if (ret == RET_OK) {
        node_shape[cnode->fullname_with_scope()] = {static_cast<int>(tensor_info->shape().size())};
      } else if (node_shape_size.find(cnode->input(1)->fullname_with_scope()) != node_shape_size.end()) {
        node_shape[cnode->fullname_with_scope()] = {node_shape_size[cnode->input(1)->fullname_with_scope()]};
      }
    } else if (prim_type == schema::PrimitiveType_StridedSlice) {
      node_shape[cnode->fullname_with_scope()] = node_shape[cnode->input(1)->fullname_with_scope()];
    } else if (prim_type == schema::PrimitiveType_Stack) {
      auto shape = node_shape[cnode->input(1)->fullname_with_scope()];
      shape.insert(shape.begin(), cnode->inputs().size() - 1);
      node_shape[cnode->fullname_with_scope()] = shape;
    }

    // Get in node shape size
    std::vector<int> in_shape_sizes;
    const size_t kShapeIndex = 2;
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      int in_shape_size = 0;
      if (utils::isa<CNodePtr>(cnode->input(i))) {
        in_shape_size = node_shape_size[cnode->input(i)->fullname_with_scope()];
        if (prim_type == schema::PrimitiveType_Reshape && i == kShapeIndex &&
            node_shape.find(cnode->input(i)->fullname_with_scope()) != node_shape.end()) {
          in_shape_size = node_shape[cnode->input(i)->fullname_with_scope()].at(0);
        }
      } else {
        tensor::TensorPtr tensor_info;
        auto ret = GetTensorInfoFromAbstract(&tensor_info, cnode, i);
        if (ret == RET_OK) {
          in_shape_size = tensor_info->shape().size();
          if (prim_type == schema::PrimitiveType_Reshape && i == kShapeIndex) {
            in_shape_size = tensor_info->shape().at(0);
          }
        }
      }
      in_shape_sizes.emplace_back(in_shape_size);
    }
    // Cal shape size infer function
    auto shape_size = shape_size_infer_registry_.at(prim_type)(in_shape_sizes, *prim_t);
    // Update node shape size map
    node_shape_size[cnode->fullname_with_scope()] = shape_size;
  }
  return node_shape_size;
}

NormFusion::NormFusion(const std::string &name, bool multigraph) : PatternProcessPass(name, multigraph) {
  input_ = std::make_shared<Var>();
  mean1_ = std::make_shared<Var>();
  mean1_axes_ = std::make_shared<Var>();
  mean2_ = std::make_shared<Var>();
  mean2_axes_ = std::make_shared<Var>();
  gamma_ = std::make_shared<Var>();
  beta_ = std::make_shared<Var>();
  epsilon_ = std::make_shared<Var>();

  shape_size_infer_registry_[schema::PrimitiveType_Activation] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_AddFusion] = BroadcastShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_BiasAdd] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Stack] = StackSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Cast] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Concat] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_ExpandDims] = ExpandDimsShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Fill] = FillShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_LayerNormFusion] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_MatMul] = MatMulShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_MulFusion] = BroadcastShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_OneHot] = OneHotSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_ReduceFusion] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Reshape] = ReShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Shape] = ShapeOpSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_SliceFusion] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Softmax] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Squeeze] = SqueezeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_StridedSlice] = StridedSliceShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Transpose] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_TileFusion] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_SquaredDifference] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_Rsqrt] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_SubFusion] = BroadcastShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_PadFusion] = CommonShapeSizeInfer;
  shape_size_infer_registry_[schema::PrimitiveType_PowFusion] = CommonShapeSizeInfer;
}

const AnfNodePtr NormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                     const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_LOG(DEBUG) << "layer_norm_fusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto add2_cnode = node->cast<CNodePtr>();
  float epsilon = 0.0f;
  int begin_norm_axis = 0;
  int begin_params_axis = 0;
  schema::PrimitiveType type = schema::PrimitiveType_NONE;
  if (!CheckPattern(func_graph, equiv, &type, &epsilon, &begin_norm_axis, &begin_params_axis)) {
    return nullptr;
  }
  auto norm_cnode = CreateNormNode(func_graph, equiv, type, epsilon, begin_norm_axis, begin_params_axis);
  if (norm_cnode == nullptr) {
    MS_LOG(DEBUG) << "create norm cnode failed";
    return nullptr;
  }
  norm_cnode->set_abstract(add2_cnode->abstract()->Clone());
  if (type == schema::PrimitiveType_LayerNormFusion) {
    norm_cnode->set_fullname_with_scope("layer_norm_" + add2_cnode->fullname_with_scope());
    MS_LOG(DEBUG) << "layer_norm node:" << norm_cnode->fullname_with_scope() << " fusion success";
  } else if (type == schema::PrimitiveType_InstanceNorm) {
    norm_cnode->set_fullname_with_scope("instance_norm_" + add2_cnode->fullname_with_scope());
    MS_LOG(DEBUG) << "instance_norm node:" << norm_cnode->fullname_with_scope() << " fusion success";
  }
  return norm_cnode;
}

const BaseRef TfNormFusion::DefinePattern() const {
  VectorRef mean1_ref = VectorRef({mean1_, input_, mean1_axes_});
  auto squared_diffference1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSquaredDifference>);
  VectorRef squared_diffference1_ref = VectorRef({squared_diffference1, input_, mean1_ref});
  auto mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mean2_ref = VectorRef({mean2_, squared_diffference1_ref, mean2_axes_});
  auto add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add1_ref = VectorRef({add1, mean2_ref, epsilon_});
  auto rsqrt1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRsqrt>);
  VectorRef rsqrt1_ref = VectorRef({rsqrt1, add1_ref});
  auto mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul2_ref = VectorRef({mul2, rsqrt1_ref, gamma_});
  VectorRef mul1_ref = VectorRef({mul1, input_, mul2_ref});
  auto mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul3_ref = VectorRef({mul3, mean1_ref, mul2_ref});
  auto sub1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  VectorRef sub1_ref = VectorRef({sub1, beta_, mul3_ref});
  auto add2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add2_ref = VectorRef({add2, mul1_ref, sub1_ref});
  return add2_ref;
}

const BaseRef OnnxLayerNormFusion::DefinePattern() const {
  VectorRef mean1_ref = VectorRef({mean1_, input_, mean1_axes_});
  VectorRef sub1_ref =
    VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>), input_, mean1_ref});
  VectorRef sub2_ref =
    VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>), input_, mean1_ref});
  VectorRef pow_ref =
    VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPowFusion>), sub2_ref, std::make_shared<Var>()});
  VectorRef mean2_ref = VectorRef({mean2_, pow_ref, mean2_axes_});
  VectorRef add1_ref =
    VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>), mean2_ref, epsilon_});
  VectorRef sqrt_ref = VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>), add1_ref});
  VectorRef div_ref =
    VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>), sub1_ref, sqrt_ref});
  VectorRef mul_ref = VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>), gamma_, div_ref});
  VectorRef add2_ref = VectorRef({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>), mul_ref, beta_});
  return add2_ref;
}
}  // namespace opt
}  // namespace mindspore
