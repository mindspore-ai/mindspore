/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fisson/fisson_util.h"
#include <unordered_set>
#include <memory>
#include "base/core_ops.h"
#include "src/common/utils.h"
#include "ops/split_with_overlap.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "ops/concat.h"
#include "tools/optimizer/parallel/spliter.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "nnacl/op_base.h"

using mindspore::converter::FmkType;
namespace mindspore {
namespace opt {
std::vector<int64_t> GetSplitPadList(const std::shared_ptr<ops::Conv2DFusion> &ori_conv_prim, int64_t input_h,
                                     int64_t input_w) {
  if (ori_conv_prim->get_pad_mode() != SAME) {
    return ori_conv_prim->get_pad_list();
  }
  int64_t output_h = static_cast<int64_t>(
    std::ceil(static_cast<float>(input_h) / static_cast<float>(ori_conv_prim->get_stride().at(kIndexH))));
  int64_t output_w = static_cast<int64_t>(
    std::ceil(static_cast<float>(input_w) / static_cast<float>(ori_conv_prim->get_stride().at(kIndexW))));

  std::vector<int64_t> new_pad_list;
  int64_t pad_up = 0, pad_down = 0, pad_left = 0, pad_right = 0;
  int64_t pad_h_all = (output_h - 1) * ori_conv_prim->get_stride().at(kIndexH) +
                      (ori_conv_prim->get_kernel_size().at(kIndexH) - 1) * ori_conv_prim->get_dilation().at(kIndexH) +
                      1 - input_h;
  int64_t pad_w_all = (output_w - 1) * ori_conv_prim->get_stride().at(kIndexW) +
                      (ori_conv_prim->get_kernel_size().at(kIndexW) - 1) * ori_conv_prim->get_dilation().at(kIndexW) +
                      1 - input_w;
  if (pad_h_all >= 0) {
    pad_up = pad_h_all / 2;
    pad_down = pad_h_all - pad_up;
  }
  new_pad_list.push_back(pad_up);
  new_pad_list.push_back(pad_down);
  if (pad_w_all >= 0) {
    pad_left = pad_w_all / 2;
    pad_right = pad_w_all - pad_left;
  }
  new_pad_list.push_back(pad_left);
  new_pad_list.push_back(pad_right);
  return new_pad_list;
}

namespace {
bool CalSplitOutputShape(int64_t splited_axis_value, const SplitInfo *split_info,
                         std::vector<int64_t> *split_axis_out_shape,
                         std::vector<int64_t> *split_axis_reduce_out_shape) {
  // ori ratio
  int64_t split_num = split_info->out_num;
  int64_t split_len = 0;
  for (int64_t i = 0; i < split_num; i++) {
    split_len += split_info->size_splits[i];
  }
  if (split_len > splited_axis_value) {
    return false;
  }
  // out-shape after splited
  int64_t tmp_value = 0;
  for (int64_t i = 0; i < split_num - 1; i++) {
    int64_t tmp = UP_DIV(split_info->size_splits[i] * splited_axis_value, split_len);
    tmp_value += tmp;
    split_axis_out_shape->push_back(tmp);
    split_axis_reduce_out_shape->push_back(tmp_value);
  }
  split_axis_out_shape->push_back(splited_axis_value - tmp_value);
  split_axis_reduce_out_shape->push_back(splited_axis_value);
  return true;
}

void CalSplitInShape(const std::vector<std::vector<ShapeVector>> &node_in_out_shapes, const SplitInfo *split_info,
                     const std::shared_ptr<ops::Conv2DFusion> &ori_conv_prim, int64_t index_node,
                     std::vector<std::vector<int64_t>> *split_axis_inputs_shape,
                     std::vector<std::vector<int64_t>> *split_axis_reduce_inputs_shape) {
  auto in_out_shape = node_in_out_shapes.at(index_node);
  auto in_shape = in_out_shape.front();
  int64_t input_h = in_shape.at(kAxisH);
  int64_t input_w = in_shape.at(kAxisW);
  auto new_pad_list = GetSplitPadList(ori_conv_prim, input_h, input_w);
  ori_conv_prim->set_pad_list(new_pad_list);
  int64_t split_num = split_info->out_num;
  int64_t tmp = 0;
  std::vector<int64_t> split_axis_shape;
  std::vector<int64_t> split_axis_reduce_shape;
  // iter splited_num
  for (int64_t index = 0; index < split_num; index++) {
    // shape
    if (split_info->axis == CuttingStragedy::CUT_H) {  // H
      if (index == 0) {
        tmp = ori_conv_prim->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[index_node][index] - 1) -
              ori_conv_prim->get_pad_list()[kPadUp] + ori_conv_prim->get_kernel_size()[kIndexH];
      } else if (index == split_num - 1) {
        tmp = ori_conv_prim->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[index_node][index] - 1) -
              ori_conv_prim->get_pad_list()[kPadDown] + ori_conv_prim->get_kernel_size()[kIndexH];
      } else {
        tmp = ori_conv_prim->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[index_node][index] - 1) - 0 +
              ori_conv_prim->get_kernel_size()[kIndexH];
      }
    }
    split_axis_shape.push_back(tmp);

    // reduce shape
    if (split_info->axis == CuttingStragedy::CUT_H) {  // H
      if (index == split_num - 1) {
        tmp = ori_conv_prim->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[index_node][index] - 1) -
              ori_conv_prim->get_pad_list()[kPadDown] - ori_conv_prim->get_pad_list()[kPadUp] +
              ori_conv_prim->get_kernel_size()[kIndexH];
      } else {
        tmp = ori_conv_prim->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[index_node][index] - 1) -
              ori_conv_prim->get_pad_list()[kPadUp] + ori_conv_prim->get_kernel_size()[kIndexH];
      }
    }
    split_axis_reduce_shape.push_back(tmp);
  }
  split_axis_inputs_shape->push_back(split_axis_shape);
  split_axis_reduce_inputs_shape->push_back(split_axis_reduce_shape);
}
}  // namespace

bool IsConv2D(const AnfNodePtr &node) {
  return (CheckPrimitiveType(node, prim::kPrimConv2D) || CheckPrimitiveType(node, prim::kPrimConv2DFusion));
}

std::shared_ptr<ops::Conv2DFusion> CopyConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_conv_prim) {
  auto new_prim = std::make_shared<ops::Conv2DFusion>();
  new_prim->set_pad(ori_conv_prim->get_pad());
  new_prim->set_in_channel(ori_conv_prim->get_in_channel());
  new_prim->set_out_channel(ori_conv_prim->get_out_channel());
  new_prim->set_dilation(ori_conv_prim->get_dilation());
  new_prim->set_format(ori_conv_prim->get_format());
  new_prim->set_group(ori_conv_prim->get_group());
  new_prim->set_kernel_size(ori_conv_prim->get_kernel_size());
  if (ori_conv_prim->get_pad_mode() == SAME) {
    new_prim->set_pad_mode(PAD);
  } else {
    new_prim->set_pad_mode(ori_conv_prim->get_pad_mode());
  }

  new_prim->set_stride(ori_conv_prim->get_stride());
  new_prim->set_activation_type(ori_conv_prim->get_activation_type());
  new_prim->set_pad_list(ori_conv_prim->get_pad_list());
  if (ori_conv_prim->GetAttr(ops::kIsDepthWise) != nullptr) {
    bool is_depth_wise = GetValue<bool>(ori_conv_prim->GetAttr(ops::kIsDepthWise));
    new_prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(is_depth_wise));
  }
  return new_prim;
}

bool UpdateSplitInfo(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &conv_nodes, SplitInfo *split_info) {
  if (split_info->axis != CuttingStragedy::CUT_H) {
    return false;
  }
  auto splited_axis = split_info->axis;
  // need to check
  if (split_info->fmk_type == FmkType::kFmkTypeCaffe ||
      split_info->fmk_type == FmkType::kFmkTypeOnnx) {  // NHWC -> NCHW
    splited_axis += 1;
  }

  size_t node_size = conv_nodes.size();
  size_t index_node = 0;
  std::vector<std::vector<ShapeVector>> node_in_out_shapes;
  while (index_node < node_size) {
    // [conv3, conv2, conv1] conv1->conv2->conv3
    auto out_node_name = conv_nodes[index_node]->fullname_with_scope();
    auto output_shapes = Spliter::GetInstance()->graph_node_output_shapes()[out_node_name];
    auto input_shapes = Spliter::GetInstance()->graph_node_input_shapes()[out_node_name];
    // 0-> in-shape 1->out-shape
    // only one in and one output
    node_in_out_shapes.push_back({input_shapes.front(), output_shapes.front()});
    index_node++;
  }

  int64_t splited_axis_value = node_in_out_shapes[0][1][splited_axis];
  int64_t final_split_axis_value = node_in_out_shapes[node_size - 1][0][splited_axis];
  split_info->ori_split_axis_value = final_split_axis_value;
  size_t split_num = split_info->size_splits.size();
  std::vector<int64_t> split_axis_out_shape;
  std::vector<int64_t> split_axis_reduce_out_shape;
  if (!CalSplitOutputShape(splited_axis_value, split_info, &split_axis_out_shape, &split_axis_reduce_out_shape)) {
    return false;
  }
  // infer in-shape after splited
  std::vector<std::vector<int64_t>> split_axis_inputs_shape{split_axis_out_shape};
  std::vector<std::vector<int64_t>> split_axis_reduce_inputs_shape{split_axis_reduce_out_shape};
  index_node = 0;
  // iter node
  while (index_node < node_size) {
    auto conv_cnode = conv_nodes[index_node]->cast<CNodePtr>();
    auto ori_conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(conv_cnode->input(kAnfPrimitiveIndex));
    CalSplitInShape(node_in_out_shapes, split_info, ori_conv_prim, index_node, &split_axis_inputs_shape,
                    &split_axis_reduce_inputs_shape);
    index_node++;
  }

  // update ratio
  split_info->size_splits.clear();
  split_info->extend_top.clear();
  split_info->extend_bottom.clear();

  int32_t top = 0;
  int32_t bottom = 0;
  split_info->size_splits.push_back(split_axis_inputs_shape[node_size][0]);
  split_info->extend_top.push_back(top);
  split_info->extend_bottom.push_back(bottom);

  for (size_t i = 1; i < split_num; i++) {
    auto begin = split_axis_reduce_inputs_shape[node_size][i] - split_axis_inputs_shape[node_size][i] + 1;
    top = split_axis_reduce_inputs_shape[node_size][i - 1] - begin + 1;
    auto value = split_axis_inputs_shape[node_size][i] - top;
    split_info->size_splits.push_back(value);
    split_info->extend_top.push_back(top);
    split_info->extend_bottom.push_back(bottom);
  }
  return true;
}

void GetMultipleOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                 std::vector<AnfNodePtr> *outputs) {
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(cnode)) {
    return;
  }
  for (size_t i = 0; i < output_num; i++) {
    auto index = NewValueNode(SizeToInt(i));
    if (CheckIfValueNodeIsNull(index)) {
      return;
    }
    size_t temp = SizeToInt(i);
    auto imm = std::make_shared<Int32Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    index->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, index});
    if (CheckIfCNodeIsNull(tuple_getitem)) {
      return;
    }
    tuple_getitem->set_fullname_with_scope(cnode->fullname_with_scope() + "_TupleGetItem_" + std::to_string(i + 1));
    outputs->push_back(tuple_getitem);
  }
}

AnfNodePtr CreateOutputsOfConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &conv_cnode,
                                 const std::vector<AnfNodePtr> &conv_outputs, SplitInfo *split_info,
                                 const std::string &node_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(conv_cnode);

  int32_t nodes_num = conv_outputs.size();
  if (nodes_num != static_cast<int32_t>(split_info->out_num)) {
    MS_LOG(ERROR) << "Conv outputs has wrong input size";
    return nullptr;
  }

  auto concat_prim = std::make_shared<ops::Concat>();
  concat_prim->set_axis(split_info->axis);

  // the inputs of concate are from the outputs of conv
  std::vector<AnfNodePtr> concate_inputs = {NewValueNode(concat_prim)};
  for (int32_t i = 0; i < nodes_num; i++) {
    concate_inputs.push_back(conv_outputs[i]);
  }

  auto concate_cnode = func_graph->NewCNode(concate_inputs);
  MS_EXCEPTION_IF_NULL(concate_cnode);

  concate_cnode->set_fullname_with_scope(node_name + "_Concat");
  concate_cnode->set_scope(conv_cnode->scope());
  std::vector<AnfNodePtr> outputs;
  GetMultipleOutputsOfAnfNode(func_graph, concate_cnode, 1, &outputs);
  return concate_cnode;
}

void CreateOutputsOfSplitWithOverlap(const FuncGraphPtr &func_graph, const AnfNodePtr &conv_node,
                                     std::vector<AnfNodePtr> *split_outputs, SplitInfo *split_info,
                                     const std::string &node_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(conv_node);
  // attr of split
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();
  split_prim->set_split_dim(split_info->axis);
  split_prim->set_number_split(split_info->out_num);
  split_prim->set_ratio(split_info->size_splits);
  split_prim->set_extend_top(split_info->extend_top);
  split_prim->set_extend_bottom(split_info->extend_bottom);
  auto conv_cnode = conv_node->cast<CNodePtr>();

  // the inputs of split is from the inputs of conv
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(split_prim)};

  // this conv only has one input, which has been ensured before
  split_inputs.push_back(conv_cnode->input(1));

  auto split_cnode = func_graph->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split_cnode);

  split_cnode->set_fullname_with_scope(node_name + "_Split");
  // create outputs op split
  GetMultipleOutputsOfAnfNode(func_graph, split_cnode, split_info->out_num, split_outputs);

  AbstractBasePtrList ptr_list;
  for (int64_t i = 0; i < split_info->out_num; i++) {
    auto node = (*split_outputs)[i];
    // set date_type same with weight
    auto type_id = static_cast<TypeId>(kNumberTypeFloat32);
    auto type_ptr = TypeIdToType(type_id);
    std::vector<int64_t> shape_vector;
    auto value_node = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    ptr_list.push_back(value_node);
  }
  split_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(ptr_list));
}

void UpdateRatioWithPadStride(int64_t *ratio, size_t split_size, int split_dim_size, int pad, int stride) {
  if (stride == 0) {
    return;
  }

  int total_block_count = 0;
  for (size_t i = 0; i < split_size; i++) {
    total_block_count += ratio[i];
  }

  std::vector<int64_t> new_ratio(split_size);
  int visited_block = 0;
  for (size_t i = 0; i < split_size - 1; i++) {
    visited_block += ratio[i];
    int cur_border = UP_DIV(split_dim_size * visited_block, total_block_count);
    new_ratio[i + 1] = cur_border;
  }

  for (size_t i = 0; i < split_size; i++) {
    ratio[i] = new_ratio[i];
  }
  return;
}
}  // namespace opt
}  // namespace mindspore
