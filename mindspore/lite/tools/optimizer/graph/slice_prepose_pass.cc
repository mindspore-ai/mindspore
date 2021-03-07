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
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include <vector>
#include <memory>
#include <set>
#include <algorithm>
#include "ops/fusion/full_connection.h"
#include "ops/reshape.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/softmax.h"
#include "ops/op_utils.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "backend/optimizer/common/helper.h"
#include "src/common/log_adapter.h"

namespace mindspore::opt {
namespace {
const int kArithmeticInputNum = 2;
const int SliceBeginIndex = 2;
const int SliceSizeIndex = 3;
int node_name_index = 0;
std::vector<int> GetSliceBeginAndSize(const CNodePtr &cnode, const int index) {
  MS_ASSERT(cnode != nullptr);
  std::vector<int> content;
  if (index != SliceBeginIndex && index != SliceSizeIndex && cnode->size() != 4) {
    return content;
  }
  auto node = cnode->input(index);
  if (node == nullptr) {
    return content;
  }
  auto paramter_node = node->cast<ParameterPtr>();
  if (paramter_node == nullptr || !paramter_node->has_default() || paramter_node->default_param() == nullptr) {
    return content;
  }
  auto paramter_value = paramter_node->default_param()->cast<ParamValueLitePtr>();
  if (paramter_value == nullptr) {
    return content;
  }
  content.resize(paramter_value->tensor_shape_size());
  if (memcpy_s(content.data(), paramter_value->tensor_shape_size(), paramter_value->tensor_addr(),
               paramter_value->tensor_shape_size()) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return {};
  }
  return content;
}

std::vector<int64_t> GetCNodeInputShape(const CNodePtr &cnode, size_t index = 1) {
  MS_ASSERT(cnode != nullptr);
  std::vector<int64_t> empty_shape;
  if (index < 1 || cnode->inputs().size() <= index) {
    MS_LOG(ERROR) << "out of index";
    return empty_shape;
  }
  auto abstract = GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract of CNode is nullptr";
    return empty_shape;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "abstract is not AbstractTensor";
    return empty_shape;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  MS_ASSERT(abstract_tensor != nullptr && abstract_tensor->shape() != nullptr);
  return abstract_tensor->shape()->shape();
}

std::vector<int64_t> GetDefaultParamShape(const ParameterPtr &param) {
  MS_ASSERT(param != nullptr);
  MS_ASSERT(param->has_default());
  std::vector<int64_t> shape_vector;
  auto default_param = param->default_param();
  if (default_param == nullptr) {
    MS_LOG(ERROR) << "default_param is nullptr";
    return shape_vector;
  }
  if (!utils::isa<ParamValueLitePtr>(default_param)) {
    MS_LOG(ERROR) << "default_param is not ParamValueLite";
    return shape_vector;
  }
  auto param_value_lite = utils::cast<ParamValueLitePtr>(default_param);
  auto shape = param_value_lite->tensor_shape();
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                 [](const int val) { return static_cast<int64_t>(val); });
  return shape_vector;
}

bool IsScalarNode(const AnfNodePtr &nodePtr) {
  if (utils::isa<ParameterPtr>(nodePtr) && nodePtr->cast<ParameterPtr>()->has_default()) {
    auto tensor = utils::cast<ParamValueLitePtr>(utils::cast<ParameterPtr>(nodePtr)->default_param());
    auto shape = tensor->tensor_shape();
    if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<mindspore::ops::SliceFusion> GetSlice(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  return GetValueNode<std::shared_ptr<mindspore::ops::SliceFusion>>(cnode->input(0));
}

std::shared_ptr<mindspore::ops::Softmax> GetSoftmax(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  return GetValueNode<std::shared_ptr<mindspore::ops::Softmax>>(cnode->input(0));
}

std::shared_ptr<mindspore::ops::Reshape> GetReshape(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  return GetValueNode<std::shared_ptr<mindspore::ops::Reshape>>(cnode->input(0));
}

std::shared_ptr<mindspore::ops::FullConnection> GetFc(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  return GetValueNode<std::shared_ptr<mindspore::ops::FullConnection>>(cnode->input(0));
}

std::vector<int> GetTransposePerm(const CNodePtr &node) {
  MS_ASSERT(node != nullptr);
  std::vector<int> perm;
  if (!CheckPrimitiveType(node, prim::kPrimTranspose)) {
    return perm;
  }
  if (node->inputs().size() != 3) {
    return perm;
  }
  auto perm_node = node->input(2);
  if (!utils::isa<ParameterPtr>(perm_node)) {
    return perm;
  }
  auto perm_param = perm_node->cast<ParameterPtr>();
  if (!perm_param->has_default() || perm_param->default_param() == nullptr) {
    return perm;
  }
  auto perm_value = perm_param->default_param()->cast<ParamValueLitePtr>();
  if (perm_value == nullptr) {
    return perm;
  }
  perm.resize(perm_value->tensor_shape()[0]);
  if (memcpy_s(perm.data(), perm_value->tensor_size(), perm_value->tensor_addr(), perm_value->tensor_size()) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    return {};
  }
  return perm;
}
}  // namespace

void SlicePreposePass::ClearCNodeAbstractValue(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto abstract = cnode->abstract();
  MS_ASSERT(abstract != nullptr);
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract of cnode is not abstract tensor, " << cnode->fullname_with_scope();
  }
  abstract->set_value(std::make_shared<AnyValue>());
}

STATUS SlicePreposePass::SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                              const CNodePtr &preceed_cnode, const int index,
                                              const TransactionPtr &tr) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(preceed_cnode != nullptr);
  if (slice_cnode->input(1) != preceed_cnode) {
    MS_LOG(ERROR) << "proceed node must be slice node's direct parent";
    return RET_ERROR;
  }
  if (IsMultiOutputTensors(graph, preceed_cnode)) {
    MS_LOG(ERROR) << "proceed node referenced by multi nodes not support swap";
    return RET_ERROR;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto node_users = manager->node_users()[slice_cnode];
  if (tr != nullptr) {  // do swap with transaction
    for (auto &node_user : node_users) {
      tr->SetEdge(node_user.first, node_user.second, preceed_cnode);
    }
    tr->SetEdge(slice_cnode, 1, preceed_cnode->input(index));
    tr->SetEdge(preceed_cnode, index, slice_cnode);
  } else {
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, preceed_cnode);
    }
    manager->SetEdge(slice_cnode, 1, preceed_cnode->input(index));
    manager->SetEdge(preceed_cnode, index, slice_cnode);
  }
  return RET_OK;
}

ValueNodePtr SlicePreposePass::CreateSliceValueNode(const FuncGraphPtr &graph, const std::vector<int64_t> &axes) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  auto new_slice = std::make_shared<mindspore::ops::SliceFusion>();
  new_slice->set_axes(axes);
  ValueNodePtr value_node = NewValueNode(new_slice);
  return value_node;
}

ValueNodePtr SlicePreposePass::CopySliceValueNode(const FuncGraphPtr &graph, const CNodePtr &slice_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  auto slice_c = GetValueNode<std::shared_ptr<mindspore::ops::SliceFusion>>(slice_cnode->input(0));
  if (slice_c == nullptr) {
    MS_LOG(ERROR) << "slice node is nullptr";
    return nullptr;
  }
  auto new_slice_c = std::make_shared<mindspore::ops::SliceFusion>();
  new_slice_c->set_axes(slice_c->get_axes());
  ValueNodePtr value_node = NewValueNode(new_slice_c);
  return value_node;
}

CNodePtr SlicePreposePass::InsertSlice(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &inputs,
                                       const CNodePtr &preceed_cnode, const int index, const TransactionPtr &tr) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(preceed_cnode != nullptr);
  auto slice_cnode = graph->NewCNode(inputs);
  slice_cnode->set_fullname_with_scope(preceed_cnode->fullname_with_scope() + "_slice_" +
                                       std::to_string(node_name_index));
  node_name_index += 1;
  tr->SetEdge(preceed_cnode, index, slice_cnode);
  return slice_cnode;
}

STATUS SlicePreposePass::VerifySliceAttrs(const CNodePtr &slice_cnode, const int dim) {
  // according to ops/slice.cc, axes >= 0, begin >= 0, size >= -1
  auto slice = GetSlice(slice_cnode);
  if (slice == nullptr) {
    MS_LOG(ERROR) << "Slice is nullptr";
    return RET_ERROR;
  }
  auto axes = slice->get_axes();
  auto begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);

  std::set<int64_t> unique_axes(axes.begin(), axes.end());
  if (axes.empty() || unique_axes.size() != axes.size()) {
    MS_LOG(DEBUG) << "Invalid slice axe attribute";
    return RET_ERROR;
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    auto axe = axes[i];
    if (dim > -1 && axe >= dim) {
      MS_LOG(ERROR) << "Invalid slice axe attribute";
      return RET_ERROR;
    }
    if (axe < 0) {
      MS_LOG(ERROR) << "Invalid slice axe attribute";
      return RET_ERROR;
    }
    if (begin[i] < 0) {  //  we not require begin[i] < ref_shape[axe], cause there may be broadcast
      MS_LOG(ERROR) << "Invalid begin input! begin[" << i << "]=" << begin[i];
      return RET_ERROR;
    }
    if (size[i] < -1) {
      MS_LOG(ERROR) << "Invalid size input! size[" << i << "]=" << size[i];
      return RET_ERROR;
    }
  }
  return RET_OK;
}

/*
 * Adjust slice's attr when broadcast happened in Arithmetic
 */
STATUS SlicePreposePass::SliceParamDeBroadcast(const CNodePtr &slice_cnode, const std::vector<int64_t> &ref_shape,
                                               std::vector<int64_t> *axes, std::vector<int> *begin,
                                               std::vector<int> *size) {
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(new_slice_cnode != nullptr);
  auto slice = GetSlice(slice_cnode);
  if (slice == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return RET_ERROR;
  }
  auto origin_axes = slice->get_axes();
  auto origin_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto origin_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  auto status = VerifySliceAttrs(slice_cnode, ref_shape.size());
  if (status != RET_OK) {
    return status;
  }
  axes->resize(ref_shape.size());
  std::iota(axes->begin(), axes->end(), 0);
  begin->assign(ref_shape.size(), 0);
  size->assign(ref_shape.size(), -1);
  bool real_slice = false;  // whether slice happened at this input
  for (size_t i = 0; i < origin_axes.size(); ++i) {
    int a = origin_axes[i];
    int b = origin_begin[i];
    int s = origin_size[i];
    int ref = ref_shape[a];
    if (ref == 1) {        // broadcast
      continue;            // sliced size is 0(such as begin=1,size=-1) is not considered.
    } else if (ref > 1) {  // not broadcast
      if (b >= ref) {
        MS_LOG(ERROR) << "slice begin[" << a << "]=" << b << ", while ref_shape[" << a << "]=" << ref << ", can't fit!";
        return RET_ERROR;
      } else {
        if (b != 0 || (s != -1 && s != ref)) {
          real_slice = true;
        }
        begin->at(a) = b;
        size->at(a) = s;
      }
    } else {  // ref == 0, not need slice
      continue;
    }
  }
  if (real_slice) {
    return lite::RET_OK;
  } else {
    return lite::RET_NO_CHANGE;
  }
}

CNodePtr SlicePreposePass::CreateReshapeCNode(const FuncGraphPtr &graph, const std::vector<int64_t> &shape_vector,
                                              const AbstractBasePtr &abstract, const CNodePtr &preceed_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  auto new_reshape = std::make_shared<mindspore::ops::Reshape>();
  if (new_reshape == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return nullptr;
  }
  ValueNodePtr value_node = NewValueNode(new_reshape);
  if (value_node == nullptr) {
    return nullptr;
  }
  std::vector<int> shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                 [](int64_t val) { return static_cast<int>(val); });
  auto shape_node = BuildIntVecParameterNode(
    graph, shape, preceed_cnode->fullname_with_scope() + "_shape_" + std::to_string(node_name_index));
  node_name_index++;
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "build parameter node failed.";
    return nullptr;
  }
  auto reshape_cnode = graph->NewCNode({value_node, preceed_cnode, shape_node});
  reshape_cnode->set_abstract(abstract);
  reshape_cnode->set_fullname_with_scope(preceed_cnode->fullname_with_scope() + "_reshape_" +
                                         std::to_string(node_name_index));
  node_name_index++;
  ClearCNodeAbstractValue(reshape_cnode);
  return reshape_cnode;
}

bool SlicePreposePass::SiblingsAreSameSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &output_node_list,
                                            const std::vector<int64_t> &ref_shape) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(output_node_list != nullptr);
  MS_ASSERT(output_node_list->size() >= 2);
  std::vector<CNodePtr> slices;
  for (auto &output_node : *(output_node_list.get())) {
    auto cnode = output_node.first->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "cnode is nullptr";
      return false;
    }
    if (!CheckPrimitiveType(cnode, prim::kPrimSliceFusion)) {
      return false;
    }
    auto slice_node = GetSlice(cnode);
    if (slice_node == nullptr) {
      MS_LOG(ERROR) << "Slice is nullptr";
      return false;
    }
    slices.push_back(cnode);
  }

  auto first_slice_cnode = slices.front();
  auto first_slice_node = GetSlice(first_slice_cnode);
  auto first_axes = first_slice_node->get_axes();
  auto first_begin = GetSliceBeginAndSize(first_slice_cnode, SliceBeginIndex);
  auto first_size = GetSliceBeginAndSize(first_slice_cnode, SliceSizeIndex);
  for (size_t i = 1; i < output_node_list->size(); ++i) {
    auto slice = GetSlice(slices[i]);
    auto axes = slice->get_axes();
    auto begin = GetSliceBeginAndSize(slices[i], SliceBeginIndex);
    auto size = GetSliceBeginAndSize(slices[i], SliceSizeIndex);
    if (axes.size() != first_axes.size()) {
      return false;
    }
    for (size_t j = 0; j < axes.size(); ++j) {
      auto axe = axes[j];
      if (!ref_shape.empty() && axe >= static_cast<int>(ref_shape.size())) {
        return false;
      }
      size_t k = 0;
      for (; k < first_axes.size(); ++k) {  // axes may not be [0...n-1], so we use nested loop to find it
        if (first_axes[k] == axe) {
          break;
        }
      }
      if (k == first_axes.size()) {
        return false;
      }
      if (begin[j] != first_begin[k]) {
        return false;
      }
      if (size[j] != first_size[k]) {
        if (ref_shape.empty()) {
          return false;
        }
        auto actual_size = size[j] > 0 ? size[j] : ref_shape[axe] - begin[j];
        auto actual_first_size = first_size[k] > 0 ? first_size[k] : ref_shape[axe] - first_begin[k];
        if (actual_size != actual_first_size) {
          return false;
        }
      }
    }
  }
  return true;
}

int64_t SlicePreposePass::GetReshapeAbnormalAxeIn(const std::vector<int64_t> &shape_in,
                                                  const std::vector<int64_t> &shape_out,
                                                  std::vector<int64_t> *mapped_axe) {
  // find shape_out's correspond axe in shape_in
  // when there are such as 3x1x1x4 => 3x1x4, mapped_axe[1] == 2
  int64_t inner_size_in = 1;
  int64_t abnormal_axe_in = -1;
  for (size_t i = 0; i < shape_in.size(); ++i) {
    inner_size_in *= shape_in[i];
    int64_t inner_size_out = 1;
    size_t j;
    for (j = 0; j < shape_out.size(); ++j) {
      inner_size_out *= shape_out[j];
      if (shape_out[j] == shape_in[i] && inner_size_out == inner_size_in) {
        mapped_axe->at(j) = i;
        break;
      }
    }
    if (j == shape_out.size() && abnormal_axe_in == -1) {
      abnormal_axe_in = i;
    }
  }
  return abnormal_axe_in;
}

int64_t SlicePreposePass::GetReshapeAbnormalIndexOut(const CNodePtr &slice_cnode,
                                                     const std::vector<int64_t> &mapped_axe,
                                                     const std::vector<int64_t> &shape_out,
                                                     std::vector<int64_t> *shape_out_copy, bool *is_normal_mode,
                                                     bool *support_abnormal_mode) {
  MS_ASSERT(slice_cnode != nullptr);
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return false;
  }
  auto slice_axes = slice_node->get_axes();
  auto slice_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto slice_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  int64_t abnormal_index_out = -1;
  for (size_t j = 0; j < shape_out.size(); ++j) {
    int index = -1;
    for (size_t i = 0; i < slice_axes.size(); ++i) {
      if (slice_axes[i] == static_cast<int64_t>(j)) {
        index = i;
        break;
      }
    }
    if (index == -1) continue;
    if (slice_begin[index] != 0 || (slice_size[index] != -1 && slice_size[index] != shape_out[j])) {
      if (mapped_axe[j] == -1) {
        if (is_normal_mode) {
          *is_normal_mode = false;
          abnormal_index_out = index;
        } else {
          *support_abnormal_mode = false;
        }
      } else {  // if there is matched axe sliced, not support abnormal mode
        shape_out_copy->at(j) =
          (slice_size[index] == -1 ? shape_out[j] - slice_begin[index] : static_cast<int64_t>(slice_size[index]));
        *support_abnormal_mode = false;
      }
    }
  }
  return abnormal_index_out;
}

bool SlicePreposePass::PreposeWithNormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                const CNodePtr &reshape_cnode, const std::vector<int64_t> &shape_in,
                                                const std::vector<int64_t> &shape_out_copy,
                                                const std::vector<int64_t> &mapped_axe) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(reshape_cnode != nullptr);
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return false;
  }
  auto slice_axes = slice_node->get_axes();
  auto slice_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto slice_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  std::vector<int64_t> new_axes(shape_in.size());
  std::iota(new_axes.begin(), new_axes.end(), 0);
  std::vector<int> new_begin(shape_in.size(), 0);
  std::vector<int> new_size(shape_in.size(), -1);

  for (size_t i = 0; i < mapped_axe.size(); ++i) {
    auto axe_in = mapped_axe[i];
    if (axe_in == -1) {
      continue;
    }
    new_begin[axe_in] = slice_begin[i];
    new_size[axe_in] = slice_size[i];
  }

  auto reshape_node = GetReshape(reshape_cnode);
  if (reshape_node == nullptr) {
    MS_LOG(ERROR) << "reshape is nullptr";
    return false;
  }
  std::vector<int> new_shape_out_copy;
  std::transform(shape_out_copy.begin(), shape_out_copy.end(), std::back_inserter(new_shape_out_copy),
                 [](int64_t val) { return static_cast<int>(val); });
  auto shape_node = BuildIntVecParameterNode(
    graph, new_shape_out_copy, reshape_cnode->fullname_with_scope() + "_shape_" + std::to_string(node_name_index));
  node_name_index++;
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "build parameter node failed.";
    return false;
  }
  reshape_cnode->set_inputs({reshape_cnode->input(0), reshape_cnode->input(1), shape_node});

  slice_node->set_axes(new_axes);
  auto new_begin_parameter = BuildIntVecParameterNode(
    graph, new_begin, slice_cnode->input(SliceBeginIndex)->cast<ParameterPtr>()->fullname_with_scope());
  auto new_size_parameter = BuildIntVecParameterNode(
    graph, new_size, slice_cnode->input(SliceSizeIndex)->cast<ParameterPtr>()->fullname_with_scope());
  slice_cnode->set_input(SliceBeginIndex, new_begin_parameter);
  slice_cnode->set_input(SliceSizeIndex, new_size_parameter);
  auto status = SwapSliceWithPreceed(graph, slice_cnode, reshape_cnode, 1);
  if (status != RET_OK) {
    return false;
  }
  reshape_cnode->set_abstract(slice_cnode->abstract()->Clone());
  ClearCNodeAbstractValue(slice_cnode);
  return true;
}

CNodePtr SlicePreposePass::CreateSlice1ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                         const CNodePtr &matmul_cnode,
                                                         const std::vector<int64_t> &shape_in,
                                                         const int64_t abnormal_axe_in,
                                                         const int64_t count_sliced_axe_in, const bool slice_at_front) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(matmul_cnode != nullptr);
  std::vector<int64_t> new_axes1(shape_in.size());
  std::iota(new_axes1.begin(), new_axes1.end(), 0);
  std::vector<int> new_begin1(shape_in.size(), 0);
  std::vector<int> new_size1(shape_in.size(), -1);
  if (slice_at_front) {
    new_begin1[abnormal_axe_in] = static_cast<int>(count_sliced_axe_in);
  } else {
    new_size1[abnormal_axe_in] = static_cast<int>(shape_in[abnormal_axe_in] - count_sliced_axe_in);
  }
  auto new_slice1 = CreateSliceValueNode(graph, new_axes1);
  if (new_slice1 == nullptr) {
    MS_LOG(ERROR) << "CreateSliceValueNode failed";
    return nullptr;
  }
  auto begin_parameter = BuildIntVecParameterNode(
    graph, new_begin1, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto size_parameter = BuildIntVecParameterNode(
    graph, new_size1, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto new_slice1_cnode = graph->NewCNode({new_slice1, matmul_cnode, begin_parameter, size_parameter});
  new_slice1_cnode->set_abstract(slice_cnode->abstract()->Clone());
  new_slice1_cnode->set_fullname_with_scope(slice_cnode->fullname_with_scope() + "_slice_" +
                                            std::to_string(node_name_index));
  node_name_index++;
  ClearCNodeAbstractValue(new_slice1_cnode);
  return new_slice1_cnode;
}

CNodePtr SlicePreposePass::CreateSlice2ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                         const CNodePtr &new_reshape1_cnode,
                                                         const std::vector<int64_t> &new_shape1,
                                                         const int64_t abnormal_axe_in,
                                                         const int64_t count_sliced_axe_in, const int64_t count_sliced2,
                                                         const bool slice_at_front) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(matmul_cnode != nullptr);
  std::vector<int64_t> new_axes2(abnormal_axe_in + 1);
  std::iota(new_axes2.begin(), new_axes2.end(), 0);
  std::vector<int> new_begin2(abnormal_axe_in + 1, 0);
  std::vector<int> new_size2(abnormal_axe_in + 1, -1);
  if (count_sliced2 > new_shape1[abnormal_axe_in]) {
    MS_LOG(WARNING) << "calculation error";
    return nullptr;
  }
  if (slice_at_front) {
    new_begin2[abnormal_axe_in] = static_cast<int>(new_shape1[abnormal_axe_in] - count_sliced2);
  } else {
    new_size2[abnormal_axe_in] = static_cast<int>(count_sliced2);
  }
  auto new_slice2 = CreateSliceValueNode(graph, new_axes2);
  if (new_slice2 == nullptr) {
    MS_LOG(ERROR) << "CreateSliceValueNode failed";
    return nullptr;
  }
  auto begin_parameter = BuildIntVecParameterNode(
    graph, new_begin2, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto size_parameter = BuildIntVecParameterNode(
    graph, new_size2, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto new_slice2_cnode = graph->NewCNode({new_slice2, new_reshape1_cnode, begin_parameter, size_parameter});
  new_slice2_cnode->set_abstract(slice_cnode->abstract()->Clone());
  new_slice2_cnode->set_fullname_with_scope(slice_cnode->fullname_with_scope() + "_slice_" +
                                            std::to_string(node_name_index));
  node_name_index++;
  ClearCNodeAbstractValue(new_slice2_cnode);
  return new_slice2_cnode;
}

bool SlicePreposePass::PreposeWithAbnormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                  const CNodePtr &reshape_cnode, const CNodePtr &matmul_cnode,
                                                  const std::vector<int64_t> &shape_in,
                                                  const std::vector<int64_t> &shape_out, const int64_t abnormal_axe_in,
                                                  const int64_t abnormal_index_out) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(reshape_cnode != nullptr);
  auto manager = graph->manager();
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return false;
  }
  auto slice_axes = slice_node->get_axes();
  auto slice_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto slice_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  auto abnormal_axe_out = slice_axes[abnormal_index_out];
  MS_ASSERT(abnormal_axe_out + 1 < shape_out.size());
  int64_t inter_size_in = 1;
  int64_t inter_size_out = 1;
  for (auto i = 0; i < abnormal_axe_in; ++i) {
    inter_size_in *= shape_in[i];
  }
  for (auto i = 0; i < abnormal_axe_out; ++i) {
    inter_size_out *= shape_out[i];
  }
  if (inter_size_in != inter_size_out) {
    MS_LOG(DEBUG) << "not support prepose now";
    return false;
  }
  int64_t outer_size_in = 1;
  int64_t outer_size_out = 1;
  for (auto i = abnormal_axe_in + 1; i < static_cast<int>(shape_in.size()); ++i) {
    outer_size_in *= shape_in[i];
  }
  for (auto i = abnormal_axe_out + 1; i < static_cast<int>(shape_out.size()); ++i) {
    outer_size_out *= shape_out[i];
  }
  const int64_t count_sliced_axe_front = slice_begin[abnormal_index_out];
  const int64_t count_sliced_axe_rear =
    slice_size[abnormal_index_out] == -1 ? 0 : (shape_out[abnormal_axe_out] - slice_size[abnormal_index_out]);
  if (count_sliced_axe_front * count_sliced_axe_rear > 0) {
    MS_LOG(DEBUG) << "not border slice at abnormal axe, prepose with reshape failed";
    return false;
  }
  bool slice_at_front = count_sliced_axe_front > 0;
  const int64_t count_sliced_out = (count_sliced_axe_front + count_sliced_axe_rear) * outer_size_out;
  const int64_t count_sliced_axe_in = count_sliced_out / outer_size_in;
  if (count_sliced_axe_in <= 0 || count_sliced_axe_in > shape_in[abnormal_axe_in]) {
    MS_LOG(DEBUG) << "amount of sliced out tensor is illegal";
    return false;
  }
  // new_slice1
  auto new_slice1_cnode = CreateSlice1ForReshapePrepose(graph, slice_cnode, matmul_cnode, shape_in, abnormal_axe_in,
                                                        count_sliced_axe_in, slice_at_front);
  if (new_slice1_cnode == nullptr) {
    return false;
  }
  // new_reshape1
  std::vector<int64_t> new_shape1(abnormal_axe_in + 1);
  for (int i = 0; i < abnormal_axe_in; ++i) {
    new_shape1[i] = shape_in[i];
  }
  new_shape1[abnormal_axe_in] = outer_size_in * (shape_in[abnormal_axe_in] - count_sliced_axe_in);
  auto new_reshape1_cnode = CreateReshapeCNode(graph, new_shape1, slice_cnode->abstract()->Clone(), new_slice1_cnode);
  if (new_reshape1_cnode == nullptr) {
    return false;
  }
  // new_slice2
  const int64_t count_sliced_abnormal_axe =
    shape_out[abnormal_axe_out] - (count_sliced_axe_front + count_sliced_axe_rear);
  const int64_t count_sliced2 = count_sliced_abnormal_axe * outer_size_out;
  auto new_slice2_cnode =
    CreateSlice2ForReshapePrepose(graph, slice_cnode, new_reshape1_cnode, new_shape1, abnormal_axe_in,
                                  count_sliced_axe_in, count_sliced2, slice_at_front);
  if (new_slice2_cnode == nullptr) {
    return false;
  }
  // new_reshape2
  std::vector<int64_t> new_shape2(shape_out.begin(), shape_out.end());
  new_shape2[abnormal_axe_out] = count_sliced_abnormal_axe;
  auto new_reshape2_cnode = CreateReshapeCNode(graph, new_shape2, slice_cnode->abstract()->Clone(), new_slice2_cnode);
  if (new_reshape2_cnode == nullptr) {
    return false;
  }
  new_reshape2_cnode->set_abstract(slice_cnode->abstract()->Clone());
  auto node_users = manager->node_users()[slice_cnode];
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, new_reshape2_cnode);
  }
  return true;
}

bool SlicePreposePass::GetArithmeticInputInfo(const CNodePtr &arithmetic_cnode, std::vector<AnfNodePtr> *inputs,
                                              std::vector<std::vector<int64_t>> *shapes,
                                              std::vector<bool> *is_default_params) {
  MS_ASSERT(arithmetic_cnode != nullptr);
  for (size_t i = 1; i < arithmetic_cnode->inputs().size(); ++i) {
    auto input = arithmetic_cnode->input(i);
    MS_ASSERT(input != nullptr);
    std::vector<int64_t> shape;
    if (utils::isa<ParameterPtr>(input)) {
      auto parameter = utils::cast<ParameterPtr>(input);
      if (!parameter->has_default()) {  // if one input is input placeholder, we can't change it
        return false;
      } else {
        shape = GetDefaultParamShape(parameter);
        is_default_params->push_back(true);
      }
    } else {  // input is CNode
      if (!utils::isa<CNodePtr>(input)) {
        MS_LOG(ERROR) << "one of Arithmetic's input is not CNode";
        return false;
      }
      shape = GetCNodeInputShape(arithmetic_cnode, i);
      is_default_params->push_back(false);
    }
    inputs->push_back(input);
    shapes->push_back(shape);
  }
  return true;
}

/*
 * Prepose condition:
 *  the softmax axis is not sliced
 */
bool SlicePreposePass::PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                          const CNodePtr &softmax_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(softmax_cnode != nullptr);
  auto softmax_node = GetSoftmax(softmax_cnode);
  if (softmax_node == nullptr) {
    MS_LOG(ERROR) << "softmax is nullptr";
    return false;
  }
  std::vector<int64_t> softmax_axis{-1};
  if (softmax_node->GetAttr(ops::kAxis) != nullptr) {
    softmax_axis = softmax_node->get_axis();
  }
  if (softmax_axis.size() != 1) {
    MS_LOG(ERROR) << "softmax axis is not a value, which don't support.";
    return false;
  }
  auto shape = GetCNodeInputShape(softmax_cnode, 1);
  if (softmax_axis.front() == -1) {
    if (shape.empty()) {  // when softmax axis == -1, shape info is needed to determine whether slice can be preposed
      return false;
    }
    softmax_axis[0] += shape.size();
  }

  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    return false;
  }
  auto slice_axes = slice_node->get_axes();
  auto slice_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto slice_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);

  for (size_t i = 0; i < slice_axes.size(); ++i) {
    if (slice_axes[i] == softmax_axis.front()) {
      if (slice_begin[i] != 0) {
        return false;
      }
      if (slice_size[i] != -1) {
        if (shape.empty() || slice_axes[i] >= static_cast<int>(shape.size())) {
          return false;
        }
        if (slice_size[i] < shape[slice_axes[i]]) {
          return false;
        }
      }
    }
  }
  auto status = SwapSliceWithPreceed(graph, slice_cnode, softmax_cnode, 1);
  if (status != RET_OK) {
    return false;
  }
  softmax_cnode->set_abstract(slice_cnode->abstract()->Clone());
  ClearCNodeAbstractValue(slice_cnode);
  return true;
}

/*
 * Prepose condition:
 *  require shape info
 *  when reshape is normal(memory view is not changed, such as 4x5 reshaped to 4x1x5), can always prepose
 *  when reshape is abnormal(such as 4x5 reshaped to 5x4), can prepose under some constraint
 * For abnormal mode:
 *  we only support border(not slice at center) slice at first mismatch axe,
 *  and we only support matmul->reshape->slice => matmul->slice->reshape*->slice*(drop "dead" data)->reshape now,
 *  cause the performance influence introduced by additional (reshape*->slice*) has not been fully evaluated.
 */
bool SlicePreposePass::PreposeWithReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                          const CNodePtr &reshape_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(reshape_cnode != nullptr);
  auto shape_in = GetCNodeInputShape(reshape_cnode, 1);
  auto shape_out = GetCNodeInputShape(slice_cnode, 1);
  auto shape_out_copy = shape_out;
  if (shape_in.empty() || shape_out.empty()) {
    MS_LOG(DEBUG) << "Reshape can't be preposed if either input or output shape is unknown";
    return false;
  }
  if (reshape_cnode->inputs().size() == 3 && utils::isa<ParameterPtr>(reshape_cnode->input(2))) {
    auto reshape_input_shape = utils::cast<ParameterPtr>(reshape_cnode->input(2));
    if (!reshape_input_shape->has_default()) {
      MS_LOG(ERROR) << "Reshape input shape is not constant";
      return false;
    }
  }
  std::vector<int64_t> mapped_axe(shape_out.size(), -1);
  int64_t abnormal_axe_in = GetReshapeAbnormalAxeIn(shape_in, shape_out, &mapped_axe);
  bool is_normal_mode = true;         // if all sliced axe can be found in input shape, normal
  bool support_abnormal_mode = true;  // if first mismatch axe are sliced and no more other axes are sliced, abnormal
  int64_t abnormal_index_out = GetReshapeAbnormalIndexOut(slice_cnode, mapped_axe, shape_out, &shape_out_copy,
                                                          &is_normal_mode, &support_abnormal_mode);
  if (is_normal_mode) {
    return PreposeWithNormalReshape(graph, slice_cnode, reshape_cnode, shape_in, shape_out_copy, mapped_axe);
  } else if (support_abnormal_mode) {
    auto matmul_node = reshape_cnode->input(1);
    MS_ASSERT(matmul_node != nullptr);
    if (IsMultiOutputTensors(graph, matmul_node) || !utils::isa<CNodePtr>(matmul_node)) {
      MS_LOG(DEBUG) << "not matmul->reshape->slice";
      return false;
    }
    auto matmul_cnode = matmul_node->cast<CNodePtr>();
    if (matmul_cnode == nullptr) {
      MS_LOG(ERROR) << "matmul_cnode is nullptr";
      return false;
    }
    if (!CheckPrimitiveType(matmul_node, prim::kPrimFullConnection) &&
        !CheckPrimitiveType(matmul_node, prim::kPrimMatMul)) {
      MS_LOG(DEBUG) << "not matmul->reshape->slice pattern";
      return false;
    }
    return PreposeWithAbnormalReshape(graph, slice_cnode, reshape_cnode, matmul_cnode, shape_in, shape_out,
                                      abnormal_axe_in, abnormal_index_out);
  }
  return false;
}

/*
 * Prepose condition:
 *  require shape info
 */
bool SlicePreposePass::PreposeWithMatmul(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const CNodePtr &matmul_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(matmul_cnode != nullptr);
  auto matmul_shape = GetCNodeInputShape(slice_cnode, 1);
  const int dims = matmul_shape.size();
  if (dims == 0) {
    // if Matmul's output shape is unknown, can't do prepose, cause we can't determine last two axes
    return false;
  }
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return RET_ERROR;
  }
  auto axes = slice_node->get_axes();
  auto begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  // matmul not support broadcast now, it makes things simpler
  auto manager = graph->manager();
  std::shared_ptr<FuncGraphTransaction> tr = std::make_shared<FuncGraphTransaction>(manager.get());
  if (tr == nullptr) {
    MS_LOG(ERROR) << "create FuncGraphTransaction failed";
    return false;
  }
  auto node_users = manager->node_users()[slice_cnode];
  bool changed = false;

  bool prepose_to_left = false;   // if only the last axe is sliced, not need prepose to left
  bool prepose_to_right = false;  // if only the second last axe is sliced, not need prepose to right
  for (size_t i = 0; i < axes.size(); ++i) {
    if (begin[i] != 0 || (size[i] != -1 && size[i] != matmul_shape[axes[i]])) {
      if (axes[i] != dims - 1) {
        prepose_to_left = true;
      } else if (axes[i] != dims - 2) {
        prepose_to_right = true;
      }
    }
  }

  if (prepose_to_left) {  //  left matrix
    auto left_axes = axes;
    auto left_begin = begin;
    auto left_size = size;
    for (size_t i = 0; i < left_axes.size(); ++i) {
      if (left_axes[i] == dims - 1) {
        left_begin[i] = 0;
        left_size[i] = -1;
      }
    }
    auto left_slice_vnode = CreateSliceValueNode(graph, left_axes);
    auto begin_parameter = BuildIntVecParameterNode(
      graph, left_begin, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
    node_name_index += 1;
    auto size_parameter = BuildIntVecParameterNode(
      graph, left_size, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
    node_name_index += 1;
    if (left_slice_vnode == nullptr) {
      MS_LOG(ERROR) << "CreateSliceValueNode failed";
      return false;
    }
    const std::vector<AnfNodePtr> inputs = {left_slice_vnode, matmul_cnode->input(1), begin_parameter, size_parameter};
    auto new_slice_cnode = InsertSlice(graph, inputs, matmul_cnode, 1, tr);
    new_slice_cnode->set_abstract(slice_cnode->abstract()->Clone());
    ClearCNodeAbstractValue(new_slice_cnode);
    changed = true;
  }
  if (prepose_to_right) {  //  right matrix
    auto right_axes = axes;
    auto right_begin = begin;
    auto right_size = size;
    for (size_t i = 0; i < right_axes.size(); ++i) {
      if (right_axes[i] == dims - 2) {
        right_begin[i] = 0;
        right_size[i] = -1;
      }
    }
    auto begin_parameter = BuildIntVecParameterNode(
      graph, right_begin, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
    node_name_index += 1;
    auto size_parameter = BuildIntVecParameterNode(
      graph, right_size, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
    node_name_index += 1;
    auto right_slice_vnode = CreateSliceValueNode(graph, right_axes);
    if (right_slice_vnode == nullptr) {
      MS_LOG(ERROR) << "CreateSliceValueNode failed";
      return false;
    }
    const std::vector<AnfNodePtr> inputs = {right_slice_vnode, matmul_cnode->input(2), begin_parameter, size_parameter};
    auto new_slice_cnode = InsertSlice(graph, inputs, matmul_cnode, 2, tr);
    new_slice_cnode->set_abstract(slice_cnode->abstract()->Clone());
    ClearCNodeAbstractValue(new_slice_cnode);
    changed = true;
  }
  if (changed) {
    matmul_cnode->set_abstract(slice_cnode->abstract()->Clone());
    for (auto &node_user : node_users) {
      tr->SetEdge(node_user.first, node_user.second, matmul_cnode);
    }
    tr->Commit();
    // we don't need graph->DropNode(slice_cnode);
  }
  return changed;
}

/*
 * Prepose condition:
 *  require shape info
 *  only support slice at first output axe now, and useAxis must be false
 */
bool SlicePreposePass::PreposeWithFullConnection(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                 const CNodePtr &fc_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(fc_cnode != nullptr);
  auto shape_in = GetCNodeInputShape(fc_cnode, 1);
  auto shape_out = GetCNodeInputShape(slice_cnode, 1);
  if (shape_in.empty() || shape_out.size() != 2) {
    MS_LOG(DEBUG) << "FullConnection can't be preposed if input shape is unknown or output shape is illegal";
    return false;
  }
  auto fc_node = GetFc(fc_cnode);
  if (fc_node == nullptr || (fc_node->GetAttr(ops::kUseAxis) != nullptr && fc_node->get_use_axis())) {
    MS_LOG(DEBUG) << "prepose with fc only support useAxis == false currently";
    return false;
  }
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "slice is nullptr";
    return RET_ERROR;
  }
  auto axes = slice_node->get_axes();
  auto begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] == 1) {
      if (begin[i] != 0 || (size[i] != -1 && size[i] != shape_out[1])) {
        MS_LOG(DEBUG) << "prepose with fc only support first output axe is sliced currently";
        return false;
      }
    }
  }

  std::vector<int64_t> mapped_axe(shape_out.size(), -1);
  int64_t inner_size_in = 1;
  for (size_t i = 0; i < shape_in.size(); ++i) {
    inner_size_in *= shape_in[i];
    int64_t inner_size_out = 1;
    for (size_t j = 0; j < shape_out.size(); ++j) {
      inner_size_out *= shape_out[j];
      if (shape_out[j] == shape_in[i] && inner_size_out == inner_size_in) {
        mapped_axe[j] = i;
        break;
      }
    }
  }
  if (mapped_axe[0] == -1) {
    MS_LOG(DEBUG) << "first axe in output can't find correspond input axe, can't do prepose";
    return false;
  }

  std::vector<int64_t> new_axes(shape_in.size());
  std::iota(new_axes.begin(), new_axes.end(), 0);
  std::vector<int> new_begin(shape_in.size(), 0);
  std::vector<int> new_size(shape_in.size(), -1);
  new_begin[mapped_axe[0]] = begin[0];
  new_size[mapped_axe[0]] = size[0];
  auto new_slice_vnode = CreateSliceValueNode(graph, new_axes);
  if (new_slice_vnode == nullptr) {
    MS_LOG(ERROR) << "CreateSliceValueNode failed";
    return false;
  }

  auto manager = graph->manager();
  std::shared_ptr<FuncGraphTransaction> tr = std::make_shared<FuncGraphTransaction>(manager.get());
  if (tr == nullptr) {
    MS_LOG(ERROR) << "create FuncGraphTransaction failed";
    return false;
  }
  auto begin_parameter = BuildIntVecParameterNode(
    graph, new_begin, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto size_parameter = BuildIntVecParameterNode(
    graph, new_size, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
  node_name_index += 1;
  const std::vector<AnfNodePtr> inputs = {new_slice_vnode, fc_cnode->input(1), begin_parameter, size_parameter};
  auto new_slice_cnode = InsertSlice(graph, inputs, fc_cnode, 1, tr);
  fc_cnode->set_abstract(slice_cnode->abstract()->Clone());
  new_slice_cnode->set_abstract(slice_cnode->abstract()->Clone());
  ClearCNodeAbstractValue(new_slice_cnode);

  auto node_users = manager->node_users()[slice_cnode];
  for (auto &node_user : node_users) {
    tr->SetEdge(node_user.first, node_user.second, fc_cnode);
  }
  tr->Commit();
  return true;
}

/*
 * Prepose condition:
 *  not require shape info, can always prepose
 */
bool SlicePreposePass::PreposeWithTranspose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                            const CNodePtr &transpose_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(transpose_cnode != nullptr);
  if (transpose_cnode->inputs().size() != 3) {
    MS_LOG(ERROR) << "transpose inputs size should be 3.";
    return false;
  }
  auto perm = GetTransposePerm(transpose_cnode);
  if (perm.empty()) {
    return false;
  }
  auto slice_node = GetSlice(slice_cnode);
  if (slice_node == nullptr) {
    MS_LOG(ERROR) << "GetSlicT failed";
    return false;
  }
  auto old_axes = slice_node->get_axes();
  auto old_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto old_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  auto slice_begin = GetSliceBeginAndSize(slice_cnode, SliceBeginIndex);
  auto slice_size = GetSliceBeginAndSize(slice_cnode, SliceSizeIndex);
  // perm is random shuffle of [0...n-1] according to ops/transpose.cc
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int>(i)) {
      for (size_t j = 0; j < old_axes.size(); ++j) {
        if (old_axes[j] == static_cast<int>(i)) {
          slice_begin[perm[i]] = old_begin[j];
          slice_size[perm[i]] = old_size[j];
          break;
        }
      }
    }
  }
  auto begin_parameter = BuildIntVecParameterNode(
    graph, slice_begin, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto size_parameter = BuildIntVecParameterNode(
    graph, slice_size, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
  node_name_index += 1;
  slice_cnode->set_input(SliceBeginIndex, begin_parameter);
  slice_cnode->set_input(SliceSizeIndex, size_parameter);
  auto status = SwapSliceWithPreceed(graph, slice_cnode, transpose_cnode, 1);
  if (status != RET_OK) {
    return false;
  }
  transpose_cnode->set_abstract(slice_cnode->abstract()->Clone());
  ClearCNodeAbstractValue(slice_cnode);
  return true;
}
/*
 * Prepose condition:
 *  may or may not require shape info
 */
bool SlicePreposePass::PreposeWithArithmetic(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                             const CNodePtr &arithmetic_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(arithmetic_cnode != nullptr);
  auto manager = graph->manager();
  auto node_users = manager->node_users()[slice_cnode];
  std::shared_ptr<FuncGraphTransaction> tr = std::make_shared<FuncGraphTransaction>(manager.get());
  if (tr == nullptr) {
    MS_LOG(ERROR) << "create FuncGraphTransaction failed";
    return false;
  }
  bool changed = false;
  std::vector<AnfNodePtr> inputs;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<bool> is_default_params;
  if (!GetArithmeticInputInfo(arithmetic_cnode, &inputs, &shapes, &is_default_params)) {
    return false;
  }

  for (size_t i = 1; i < arithmetic_cnode->inputs().size(); ++i) {
    auto &input = inputs[i - 1];
    if (IsScalarNode(input)) {  // scalar not need prepose
      continue;
    }
    auto &shape = shapes[i - 1];
    const size_t another_index = kArithmeticInputNum - i;
    auto &another_input = inputs[another_index];
    auto &another_shape = shapes[another_index];
    if (IsScalarNode(input)) {
      continue;
    } else if (shape.empty()) {           // infershape failed at this input
      if (IsScalarNode(another_input)) {  // if another input is scalar, we can process this one
        auto new_slice_vnode = CopySliceValueNode(graph, slice_cnode);
        if (new_slice_vnode == nullptr) {
          changed = false;
          break;
        }
        std::vector<AnfNodePtr> slice_inputs = {new_slice_vnode, arithmetic_cnode->input(i),
                                                slice_cnode->input(SliceBeginIndex),
                                                slice_cnode->input(SliceSizeIndex)};
        auto new_slice_cnode = InsertSlice(graph, slice_inputs, arithmetic_cnode, i, tr);
        new_slice_cnode->set_abstract(slice_cnode->abstract()->Clone());
        ClearCNodeAbstractValue(new_slice_cnode);
        changed = true;
        break;
      } else {  // if another input's shape is not scalar, can't be processed
        changed = false;
        break;
      }
    } else {  // shape not empty
      if (!another_shape.empty() || IsScalarNode(another_input)) {
        std::vector<int64_t> new_axes;
        std::vector<int> new_begin;
        std::vector<int> new_size;
        auto status = SliceParamDeBroadcast(slice_cnode, shape, &new_axes, &new_begin, &new_size);
        if (status == lite::RET_NO_CHANGE) {
          continue;
        }
        if (status != lite::RET_OK) {
          changed = false;
          break;
        }
        auto new_slice_vnode = CreateSliceValueNode(graph, new_axes);
        if (new_slice_vnode == nullptr) {
          changed = false;
          break;
        }
        auto begin_parameter = BuildIntVecParameterNode(
          graph, new_begin, slice_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
        node_name_index += 1;
        auto size_parameter = BuildIntVecParameterNode(
          graph, new_size, slice_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
        node_name_index += 1;
        std::vector<AnfNodePtr> slice_inputs = {new_slice_vnode, arithmetic_cnode->input(i), begin_parameter,
                                                size_parameter};
        auto new_slice_cnode = InsertSlice(graph, slice_inputs, arithmetic_cnode, i, tr);
        new_slice_cnode->set_abstract(slice_cnode->abstract()->Clone());
        ClearCNodeAbstractValue(new_slice_cnode);
        changed = true;
      } else {
        changed = false;
        break;
      }
    }
  }
  if (changed) {
    arithmetic_cnode->set_abstract(slice_cnode->abstract()->Clone());
    for (auto &node_user : node_users) {
      tr->SetEdge(node_user.first, node_user.second, arithmetic_cnode);
    }
    tr->Commit();
    // we don't need graph->DropNode(slice_cnode);
  }
  return changed;
}  // namespace mindspore::opt
/*
 * Prepose condition:
 *  not require shape info
 */
bool SlicePreposePass::MergeSequentialSlice(const FuncGraphPtr &graph, const CNodePtr &slice1_cnode,
                                            const CNodePtr &slice2_cnode) {
  if (slice2_cnode->inputs().size() != kArithmeticInputNum) {
    MS_LOG(INFO) << "Slice read attrs from input is not supported now";
    return false;
  }
  auto slice1_node = GetSlice(slice1_cnode);  // bottom node
  auto slice2_node = GetSlice(slice2_cnode);  // top node
  if (slice1_node == nullptr || slice2_node == nullptr) {
    MS_LOG(ERROR) << "slice is null";
    return false;
  }
  auto begin_slice1 = GetSliceBeginAndSize(slice1_cnode, SliceBeginIndex);
  auto size_slice1 = GetSliceBeginAndSize(slice1_cnode, SliceSizeIndex);
  auto axes_slice1 = slice1_node->get_axes();
  auto begin_slice2 = GetSliceBeginAndSize(slice2_cnode, SliceBeginIndex);
  auto size_slice2 = GetSliceBeginAndSize(slice2_cnode, SliceSizeIndex);
  auto axes_slice2 = slice2_node->get_axes();
  auto status1 = VerifySliceAttrs(slice1_cnode);
  auto status2 = VerifySliceAttrs(slice2_cnode);
  if (status1 != RET_OK || status2 != RET_OK) {
    return false;
  }

  auto manager = graph->manager();
  auto node_users = manager->node_users()[slice1_cnode];
  int64_t axe_max1 = *std::max_element(axes_slice1.begin(), axes_slice1.end());
  int64_t axe_max2 = *std::max_element(axes_slice2.begin(), axes_slice2.end());
  int64_t axe_max = std::max(axe_max1, axe_max2);
  auto begin_new = begin_slice2;
  auto size_new = size_slice2;
  auto axes_new = slice2_node->get_axes();
  axes_new.resize(axe_max + 1);
  std::iota(axes_new.begin(), axes_new.end(), 0);
  begin_new.assign(axe_max + 1, 0);
  size_new.assign(axe_max + 1, -1);
  for (int i = 0; i <= axe_max; ++i) {
    for (size_t j = 0; j < axes_slice2.size(); ++j) {
      if (axes_slice2[j] == i) {
        begin_new[i] = begin_slice2[j];
        size_new[i] = size_slice2[j];
        break;
      }
    }
    for (size_t j = 0; j < axes_slice1.size(); ++j) {
      if (axes_slice1[j] == i) {
        begin_new[i] = begin_new[i] + begin_slice1[j];
        if (size_new[i] == -1) {
          size_new[i] = size_slice1[j];
        } else {
          if (size_slice1[j] == -1) {
            size_new[i] = std::max(size_new[i] - begin_slice1[i], 0);  // clip with zero to avoid invalid negative value
          } else {
            size_new[i] = std::max(std::min(size_new[i] - begin_slice1[j], size_slice1[j]), 0);
          }
        }
        break;
      }
    }
  }
  slice2_node->set_axes(axes_new);
  auto begin_parameter = BuildIntVecParameterNode(
    graph, begin_new, slice2_cnode->fullname_with_scope() + "_begin_" + std::to_string(node_name_index));
  node_name_index += 1;
  auto size_parameter = BuildIntVecParameterNode(
    graph, size_new, slice2_cnode->fullname_with_scope() + "_size_" + std::to_string(node_name_index));
  node_name_index += 1;
  slice2_cnode->set_input(SliceBeginIndex, begin_parameter);
  slice2_cnode->set_input(SliceSizeIndex, size_parameter);
  slice2_cnode->set_abstract(slice1_cnode->abstract()->Clone());
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, slice2_cnode);
  }
  return true;
}

/*
 * Prepose condition:
 *  when all sibling slices do same work
 *  can be optimize to not require all siblings are slice
 */
bool SlicePreposePass::MergeParallelSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &slices) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slices->size() >= 2);
  auto manager = graph->manager();
  auto first_slice = utils::cast<CNodePtr>(slices->at(0).first);
  if (first_slice == nullptr || !CheckPrimitiveType(first_slice, prim::kPrimSliceFusion)) {
    MS_LOG(ERROR) << "first node is not Slice";
    return false;
  }
  auto first_parent = first_slice->input(1);
  if (first_parent == nullptr) {
    MS_LOG(ERROR) << "first slice node's parent is nullptr";
    return false;
  }
  std::shared_ptr<FuncGraphTransaction> tr = std::make_shared<FuncGraphTransaction>(manager.get());
  if (tr == nullptr) {
    MS_LOG(ERROR) << "create FuncGraphTransaction failed";
    return false;
  }
  for (size_t i = 1; i < slices->size(); ++i) {
    auto slice = utils::cast<CNodePtr>(slices->at(i).first);
    if (slice == nullptr || !CheckPrimitiveType(slice, prim::kPrimSliceFusion)) {
      MS_LOG(ERROR) << "current node is not Slice";
      return false;
    }
    auto parent = slice->input(1);
    if (parent == nullptr || parent != first_parent) {
      MS_LOG(ERROR) << "not all slices have same parent node";
      return false;
    }
    auto node_users = manager->node_users()[slices->at(i).first];
    for (auto &node_user : node_users) {
      tr->SetEdge(node_user.first, node_user.second, slices->at(0).first);
    }
  }
  tr->Commit();
  return true;
}

bool SlicePreposePass::DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                 const CNodePtr &preceed_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(preceed_cnode != nullptr);
  if (CheckPrimitiveType(preceed_cnode, prim::kPrimSoftmax)) {
    return PreposeWithSoftmax(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimReshape)) {
    return PreposeWithReshape(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimMatMul)) {
    return PreposeWithMatmul(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimFullConnection)) {
    return PreposeWithFullConnection(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimTranspose)) {
    return PreposeWithTranspose(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimSubFusion) ||
             CheckPrimitiveType(preceed_cnode, prim::kPrimMulFusion) ||
             CheckPrimitiveType(preceed_cnode, prim::kPrimAddFusion)) {
    return PreposeWithArithmetic(graph, slice_cnode, preceed_cnode);
  } else if (CheckPrimitiveType(preceed_cnode, prim::kPrimSliceFusion)) {
    return MergeSequentialSlice(graph, slice_cnode, preceed_cnode);
  }
  return false;
}

bool SlicePreposePass::Run(const FuncGraphPtr &graph) {
  if (fmk_type != lite::converter::FmkType_TF && fmk_type != lite::converter::FmkType_TFLITE) {
    MS_LOG(INFO) << "The framework type of model should be tf/tflite.";
    return false;
  }
  MS_ASSERT(graph != nullptr);
  bool changed = false;
  while (true) {
    bool this_time_changed = false;
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (node->func_graph() != graph) {
        continue;
      }
      if (!utils::isa<CNodePtr>(node) || !CheckPrimitiveType(node, prim::kPrimSliceFusion)) {
        continue;
      }
      auto slice_cnode = node->cast<CNodePtr>();
      if (!CheckIsAllInputsParam(slice_cnode)) {  // only support begin and size is const tensor.
        MS_LOG(INFO) << "SlicePrepose not support input is variable now";
        continue;
      }
      auto slice_node = GetSlice(slice_cnode);
      if (slice_node == nullptr) {
        MS_LOG(ERROR) << "slice is nullptr";
        continue;
      }
      auto preceed_node = slice_cnode->input(1);
      if (preceed_node == nullptr) {
        MS_LOG(ERROR) << "proceed node is nullptr";
        continue;
      }
      auto output_tensor_num = GetOutputTensorNum(preceed_node);
      if (output_tensor_num > 1) {
        continue;
      }
      auto output_node_list = GetRealNodeUsedList(graph, utils::cast<AnfNodePtr>(preceed_node));
      if (output_node_list->size() > 1) {  // referenced by multi nodes
        if (SiblingsAreSameSlice(graph, output_node_list)) {
          if (MergeParallelSlice(graph, output_node_list)) {
            this_time_changed = true;
            break;
          }
        }
        continue;
      } else {
        if (utils::isa<ParameterPtr>(preceed_node)) {
          /*
           * if preceed_node is parameter without default param, it's input placeholder, so we can't prepose
           * if preceed_node is parameter with default param, constant_folding will process it
           */
          continue;
        }
        auto preceed_cnode = preceed_node->cast<CNodePtr>();
        if (preceed_cnode == nullptr) {
          MS_LOG(ERROR) << "preceed_cnode is nullptr";
          continue;
        }
        if (DoPrepose(graph, slice_cnode, preceed_cnode)) {
          this_time_changed = true;
          break;
        }
      }
    }
    if (this_time_changed) {
      changed = true;
    } else {
      break;
    }
  }
  return changed;
}
}  // namespace mindspore::opt
