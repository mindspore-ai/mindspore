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

#include <unordered_set>
#include <memory>
#include "tools/optimizer/fisson/fisson_util.h"
#include "base/core_ops.h"
#include "src/common/utils.h"
#include "mindspore/core/ops/split_with_overlap.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "ops/concat.h"
#include "tools/optimizer/parallel/spliter.h"
#include "tools/optimizer/parallel/split_strategy.h"

using mindspore::lite::converter::FmkType;
namespace mindspore {
namespace opt {

namespace {

bool CalSplitOutputShape(int32_t splited_axis_value, const SplitInfo *split_info,
                         std::vector<int32_t> *split_axis_out_shape,
                         std::vector<int32_t> *split_axis_reduce_out_shape) {
  // ori ratio
  int32_t split_num = split_info->size_splits.size();
  int32_t split_len = 0;
  for (int32_t i = 0; i < split_num; i++) {
    split_len += split_info->size_splits[i];
  }
  if (split_len > splited_axis_value) {
    return false;
  }
  // out-shape after splited
  int32_t tmp_value = 0;
  for (int32_t i = 0; i < split_num - 1; i++) {
    int32_t tmp = (split_info->size_splits[i] * splited_axis_value) / split_len;
    tmp_value += tmp;
    split_axis_out_shape->push_back(tmp);
    split_axis_reduce_out_shape->push_back(tmp_value);
  }
  split_axis_out_shape->push_back(splited_axis_value - tmp_value);
  split_axis_reduce_out_shape->push_back(splited_axis_value);
  return true;
}

void CalSplitInShape(int32_t splited_axis_value, const SplitInfo *split_info,
                     const std::shared_ptr<ops::Conv2DFusion> &ori_attr, int32_t idx_node,
                     std::vector<std::vector<int32_t>> *split_axis_inputs_shape,
                     std::vector<std::vector<int32_t>> *split_axis_reduce_inputs_shape) {
  int32_t split_num = split_info->size_splits.size();
  int32_t tmp = 0;
  std::vector<int32_t> split_axis_shape;
  std::vector<int32_t> split_axis_reduce_shape;

  // iter splited_num
  for (int32_t idx = 0; idx < split_num; idx++) {
    // shape
    if (split_info->axis == CuttingStragedy::CUT_H) {  // H
      if ((splited_axis_value + ori_attr->get_pad_list()[kPadUp] + ori_attr->get_pad_list()[kPadDown] -
           (ori_attr->get_kernel_size()[kAxisH] - 1)) %
            ori_attr->get_stride()[kIndexH] ==
          0) {
        if (idx == 0) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx]) +
                (ori_attr->get_kernel_size()[kAxisH] - 1) - ori_attr->get_pad_list()[kPadUp];
        } else if (idx == split_num - 1) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx]) +
                (ori_attr->get_kernel_size()[kAxisH] - 1) - ori_attr->get_pad_list()[kPadDown];
        } else {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx]) +
                (ori_attr->get_kernel_size()[kAxisH] - 1) - 0;
        }
      } else {
        if (idx == 0) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) -
                ori_attr->get_pad_list()[kPadUp] + ori_attr->get_kernel_size()[kAxisH];
        } else if (idx == split_num - 1) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) -
                ori_attr->get_pad_list()[kPadDown] + ori_attr->get_kernel_size()[kAxisH];
        } else {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) - 0 +
                ori_attr->get_kernel_size()[kAxisH];
        }
      }

    } else if (split_info->axis == CuttingStragedy::CUT_W) {  // W
      if (idx == 0) {
        tmp = ori_attr->get_stride()[kIndexW] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) -
              ori_attr->get_pad_list()[kPadLeft] + ori_attr->get_kernel_size()[kAxisW];
      } else if (idx == split_num - 1) {
        tmp = ori_attr->get_stride()[kIndexW] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) -
              ori_attr->get_pad_list()[kPadRight] + ori_attr->get_kernel_size()[kAxisW];
      } else {
        tmp = ori_attr->get_stride()[kIndexW] * ((*split_axis_inputs_shape)[idx_node][idx] - 1) - 0 +
              ori_attr->get_kernel_size()[kAxisW];
      }
    }
    split_axis_shape.push_back(tmp);

    // reduce shape
    if (split_info->axis == CuttingStragedy::CUT_H) {  // H
      if ((splited_axis_value + ori_attr->get_pad_list()[kPadUp] + ori_attr->get_pad_list()[kPadDown] -
           (ori_attr->get_kernel_size()[kAxisH] - 1)) %
            ori_attr->get_stride()[kIndexH] ==
          0) {
        if (idx == split_num - 1) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[idx_node][idx]) +
                ori_attr->get_kernel_size()[kAxisH] - 1 - ori_attr->get_pad_list()[kPadDown] -
                ori_attr->get_pad_list()[kPadUp];
        } else {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[idx_node][idx]) +
                ori_attr->get_kernel_size()[kAxisH] - 1 - ori_attr->get_pad_list()[kPadUp];
        }
      } else {
        if (idx == split_num - 1) {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[idx_node][idx] - 1) -
                ori_attr->get_pad_list()[kPadDown] - ori_attr->get_pad_list()[kPadUp] +
                ori_attr->get_kernel_size()[kAxisH];
        } else {
          tmp = ori_attr->get_stride()[kIndexH] * ((*split_axis_reduce_inputs_shape)[idx_node][idx] - 1) -
                ori_attr->get_pad_list()[kPadUp] + ori_attr->get_kernel_size()[kAxisH];
        }
      }
    } else if (split_info->axis == CuttingStragedy::CUT_W) {  // W
      if (idx == split_num - 1) {
        tmp = ori_attr->get_stride()[kIndexW] * ((*split_axis_reduce_inputs_shape)[idx_node][idx] - 1) -
              ori_attr->get_pad_list()[kPadRight] - ori_attr->get_pad_list()[kPadLeft] +
              ori_attr->get_kernel_size()[kAxisW];
      } else {
        tmp = ori_attr->get_stride()[kIndexW] * ((*split_axis_reduce_inputs_shape)[idx_node][idx] - 1) -
              ori_attr->get_pad_list()[kPadLeft] + ori_attr->get_kernel_size()[kAxisW];
      }
    }
    split_axis_reduce_shape.push_back(tmp);
  }
  split_axis_inputs_shape->push_back(split_axis_shape);
  split_axis_reduce_inputs_shape->push_back(split_axis_reduce_shape);
}

bool CheckPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, int32_t splited_axis_value) {
  return !(splited_axis_value == ori_attr->get_kernel_size()[kAxisH] && ori_attr->get_pad_list()[kPadUp] == 0 &&
           ori_attr->get_pad_list()[kPadDown] == 0);
}
}  // namespace

bool IsConv2D(const AnfNodePtr &node) {
  return (CheckPrimitiveType(node, prim::kPrimConv2D) || CheckPrimitiveType(node, prim::kPrimConv2DFusion));
}

std::shared_ptr<ops::Conv2DFusion> CopyConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr) {
  auto prim = std::make_shared<ops::Conv2DFusion>();
  prim->set_pad(ori_attr->get_pad());
  prim->set_in_channel(ori_attr->get_in_channel());
  prim->set_out_channel(ori_attr->get_out_channel());
  prim->set_dilation(ori_attr->get_dilation());
  prim->set_format(ori_attr->get_format());
  prim->set_group(ori_attr->get_group());
  prim->set_kernel_size(ori_attr->get_kernel_size());
  prim->set_pad_mode(ori_attr->get_pad_mode());
  prim->set_pad_list(ori_attr->get_pad_list());
  prim->set_stride(ori_attr->get_stride());
  prim->set_activation_type(ori_attr->get_activation_type());
  prim->set_pad_list(prim->get_pad_list());
  prim->set_pad_mode(PAD);
  return prim;
}

bool UpdateSplitInfo(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &conv_nodes, SplitInfo *split_info) {
  if (split_info->axis != CuttingStragedy::CUT_H) {
    return false;
  }
  auto splited_axis = split_info->axis;
  if (split_info->fmk_type == FmkType::FmkType_CAFFE ||
      split_info->fmk_type == FmkType::FmkType_ONNX) {  // NHWC -> NCHW
    splited_axis += 1;
  }

  const int32_t node_size = conv_nodes.size();
  int32_t idx_node = 0;
  std::vector<std::vector<ShapeVector>> node_in_out_shapes;
  while (idx_node < node_size) {
    // [conv3, conv2, conv1] conv1->conv2->conv3
    auto out_node_name = conv_nodes[idx_node]->fullname_with_scope();
    auto output_shapes = Spliter::GetInstance()->graph_node_output_shapes()[out_node_name];
    auto input_shapes = Spliter::GetInstance()->graph_node_input_shapes()[out_node_name];
    // 0-> in-shape 1->out-shape
    // only one in and one output
    node_in_out_shapes.push_back({output_shapes.front(), input_shapes.front()});
    idx_node++;
  }

  const int32_t splited_axis_value = node_in_out_shapes[0][1][splited_axis];
  int32_t split_num = split_info->size_splits.size();
  std::vector<int32_t> split_axis_out_shape;
  std::vector<int32_t> split_axis_reduce_out_shape;
  if (!CalSplitOutputShape(splited_axis_value, split_info, &split_axis_out_shape, &split_axis_reduce_out_shape)) {
    return false;
  }
  // infer in-shape after splited
  std::vector<std::vector<int32_t>> split_axis_inputs_shape{split_axis_out_shape};
  std::vector<std::vector<int32_t>> split_axis_reduce_inputs_shape{split_axis_reduce_out_shape};
  idx_node = 0;
  // iter node
  while (idx_node < node_size) {
    auto conv_cnode = conv_nodes[idx_node]->cast<CNodePtr>();
    auto ori_attr = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(conv_cnode->input(kAnfPrimitiveIndex));
    if (!CheckPrim(ori_attr, splited_axis_value)) {
      return false;
    }
    CalSplitInShape(splited_axis_value, split_info, ori_attr, idx_node, &split_axis_inputs_shape,
                    &split_axis_reduce_inputs_shape);
    idx_node++;
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

  for (int32_t i = 1; i < split_num; i++) {
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
    auto idx = NewValueNode(SizeToInt(i));
    if (CheckIfValueNodeIsNull(idx)) {
      return;
    }
    size_t temp = SizeToInt(i);
    auto imm = std::make_shared<Int32Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
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
  // default to format khwc or nhwc
  split_prim->set_trans_format(true);

  // the inputs of split is from the inputs of conv
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(split_prim)};
  auto conv_cnode = conv_node->cast<CNodePtr>();

  // this conv only has one input, which has been ensured before
  split_inputs.push_back(conv_cnode->input(1));

  auto split_cnode = func_graph->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split_cnode);

  split_cnode->set_fullname_with_scope(node_name + "_Split");
  // create outputs op split
  GetMultipleOutputsOfAnfNode(func_graph, split_cnode, split_info->out_num, split_outputs);

  AbstractBasePtrList ptr_list;
  for (size_t i = 0; i < split_info->out_num; i++) {
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

}  // namespace opt
}  // namespace mindspore
