/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "mindspore/ccsrc/utils/dynamic_obfuscation/dynamic_obfuscation.h"
#include <cstdlib>
#include <algorithm>
#include <map>
#include <memory>
#include <functional>
#include <random>
#include "mindspore/ccsrc/utils/dynamic_obfuscation/registry_opaque_predicate.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/info.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ops/core_ops.h"

namespace mindspore {
using Tensor = mindspore::tensor::Tensor;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

constexpr int expand_rate = 10;  // total node need for a switch graph

ShapeVector get_node_shape(AnfNodePtr input_node) {
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Input_node is nullptr, get shape failed!";
    return {};
  }
  AbstractBasePtr input_abstract = input_node->abstract();
  if (input_abstract == nullptr) {
    MS_LOG(ERROR) << "The abstract of input_node is nullptr, get shape failed!";
    return {};
  }
  AbstractTensorPtr input_abstract_tensor = input_abstract->cast<mindspore::abstract::AbstractTensorPtr>();
  mindspore::abstract::ShapePtr shape_ptr = input_abstract_tensor->shape();
  return shape_ptr->shape();
}

TypeId get_node_dtype(AnfNodePtr input_node) {
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Input_node is nullptr, get dtype failed!";
    return {};
  }
  AbstractBasePtr input_abstract = input_node->abstract();
  if (input_abstract == nullptr) {
    MS_LOG(ERROR) << "The abstract of input_node is nullptr, get dtype failed!";
    return {};
  }
  AbstractTensorPtr input_abstract_tensor = input_abstract->cast<mindspore::abstract::AbstractTensorPtr>();
  AbstractBasePtr node_element = input_abstract_tensor->element();
  mindspore::abstract::AbstractScalarPtr node_element_abs =
    node_element->cast<mindspore::abstract::AbstractScalarPtr>();

  TypeId data_type = node_element_abs->BuildType()->type_id();
  return data_type;
}

std::vector<std::string> name_split(std::string &node_name, const std::string &split_sign) {
  node_name += split_sign;
  unsigned int name_len = node_name.size();
  std::string::size_type split_pos;
  std::vector<std::string> res;
  for (unsigned int i = 0; i < name_len; i++) {
    split_pos = node_name.find(split_sign, i);
    if (split_pos < name_len) {
      std::string sub_str = node_name.substr(i, split_pos - i);
      res.push_back(sub_str);
      i = split_pos + split_sign.size() - 1;
    }
  }
  return res;
}

ValueNodePtr build_tuple_value_node(std::vector<int64_t> values) {
  mindspore::ValueNodePtr v_node = std::make_shared<mindspore::ValueNode>(MakeValue(values));
  AbstractBasePtrList abs_list;
  std::transform(values.begin(), values.end(), std::back_inserter(abs_list), [](const int64 &item) {
    return std::make_shared<mindspore::abstract::AbstractScalar>(int64_t(item));
  });
  auto abs_tuple = std::make_shared<mindspore::abstract::AbstractTuple>(abs_list);
  v_node->set_abstract(abs_tuple);
  return v_node;
}

ValueNodePtr make_int_node(FuncGraphPtr func_graph, int int_value) {
  ShapeVector int_shape{1, 1};
  tensor::TensorPtr int_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeInt32, int_shape);
  int *tensor_data = reinterpret_cast<int *>(int_tensor->data_c());
  for (int i = 0; i < int_tensor->data().size(); i++) {
    tensor_data[i] = int_value;
  }
  mindspore::ValueNodePtr int_tensor_node = std::make_shared<mindspore::ValueNode>(int_tensor);
  int_tensor_node->set_abstract(int_tensor->ToAbstract());
  (void)func_graph->AddValueNode(int_tensor_node);
  return int_tensor_node;
}

tensor::TensorPtr make_weight_tensor(TypeId type_id, ShapeVector shape) {
  tensor::TensorPtr weight_tensor = std::make_shared<Tensor>(type_id, shape);
  std::default_random_engine generator;
  int max_count = 10000;
  int tensor_size = weight_tensor->data().size();
  if (type_id == kNumberTypeFloat64) {
    const double mean_64 = 0;
    const double stddev_64 = 1;
    std::normal_distribution<double> dist_64(mean_64, stddev_64);
    double *float_64_data = reinterpret_cast<double *>(weight_tensor->data_c());
    for (int i = 0; i < std::min(tensor_size, max_count); i++) {
      double random_float_64 = dist_64(generator);
      if (random_float_64 > 0) {
        float_64_data[i] = random_float_64;
      }
    }
  } else {
    MS_LOG(DEBUG) << "Type id is: " << type_id << ", weights will be float_32 format.";
    const float mean = 0;
    const float stddev = 1;
    std::normal_distribution<float> dist_32(mean, stddev);
    float *float_32_data = reinterpret_cast<float *>(weight_tensor->data_c());
    for (int i = 0; i < std::min(tensor_size, max_count); i++) {
      float random_float_32 = dist_32(generator);
      if (random_float_32 > 0) {
        float_32_data[i] = random_float_32;
      }
    }
  }
  return weight_tensor;
}

bool CheckIfObfuscated(const FuncGraphPtr &func_graph) {
  auto mgr = Manage(func_graph);
  auto all_nodes = mgr->all_nodes();
  for (AnfNodePtr node : all_nodes) {
    std::string node_name = node->fullname_with_scope();
    if (node_name.find("Switch") != std::string::npos) {
      return true;
    }
  }
  return false;
}

FuncGraphPtr DynamicObfuscator::ObfuscateMindIR(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Start obfuscation.";
  MS_EXCEPTION_IF_NULL(func_graph);
  if (CheckIfObfuscated(func_graph)) {
    MS_EXCEPTION(ValueError) << "The input model has been onfuscated, do not obfuscate it again.";
  }
  auto mgr = Manage(func_graph);
  MS_EXCEPTION_IF_NULL(mgr);
  auto all_nodes = mgr->all_nodes();
  int node_nums = all_nodes.size();
  MS_LOG(INFO) << "Total node num: " << node_nums;
  // init the number control node that has been build
  used_control_node_ = 0;
  if (obf_password_ == 0) {
    int obfuscate_target_num = std::ceil(all_nodes.size() * obf_ratio_ / expand_rate);
    int obfuscate_node_num = 0;
    // record customized_func computing results
    for (AnfNodePtr node : all_nodes) {
      std::string obf_type = single_op_obfuscate_type(node);
      MS_LOG(INFO) << "obf_type: " << obf_type;
      if (obf_type == "MatMul-op") {
        obfuscate_node_num += 1;
        MS_LOG(INFO) << "Find a MatMul Node: " << node->fullname_with_scope();
        bool customized_func_result = mindspore::kernel::CustomizedOpaquePredicate::GetInstance().run_function(
          static_cast<float>(1), static_cast<float>(1));
        customized_func_results_.push_back(customized_func_result);
      }
      if (obfuscate_node_num >= obfuscate_target_num) {
        break;
      }
    }
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().init_calling_count();
  }
  // do op-wise fake-branch obfuscation
  (void)op_wise_fake_branch(func_graph);
  if (used_control_node_ == 0) {
    MS_LOG(WARNING)
      << "The model has not been obfuscated, which means obf_password or customized_func is not need to set.";
  }
  return func_graph;
}

void DynamicObfuscator::op_wise_fake_branch(FuncGraphPtr func_graph) {
  auto mgr = Manage(func_graph);
  auto all_nodes = mgr->all_nodes();
  int obfuscate_target_num = std::ceil(all_nodes.size() * obf_ratio_ / expand_rate);
  int obfuscate_node_num = 0;
  for (AnfNodePtr node : all_nodes) {
    std::string obf_type = single_op_obfuscate_type(node);
    MS_LOG(INFO) << "The obf_type is: " << obf_type;
    if (obf_type == "MatMul-op") {
      obfuscate_node_num += 1;
      MS_LOG(INFO) << "Find a MatMul Node: " << node->fullname_with_scope();
      std::vector<AnfNodePtr> node_inputs = node->cast<mindspore::CNodePtr>()->inputs();
      mindspore::AnfNodePtr input_1 = node_inputs[1];
      CNodePtr control_c_node = get_control_node(func_graph, input_1);
      (void)replace_matmul_node(node->cast<CNodePtr>(), func_graph, control_c_node);
      MS_LOG(INFO) << "Finished replacement for: " << node->fullname_with_scope();
    }
    if (obfuscate_node_num >= obfuscate_target_num) {
      break;
    }
  }
}

std::string DynamicObfuscator::single_op_obfuscate_type(AnfNodePtr node) {
  if (node->isa<CNode>()) {
    std::string node_name = node->fullname_with_scope();
    MS_LOG(INFO) << "The node_name is: " << node_name;
    std::vector<std::string> split_words = name_split(node_name, "/");
    std::string op_name = split_words[split_words.size() - 1];
    for (std::string target_op_name : obf_target_op) {
      int op_name_len = op_name.size();
      int target_name_len = target_op_name.size();
      if ((op_name_len >= target_name_len) && (op_name.substr(0, target_name_len) == target_op_name)) {
        return target_op_name;
      }
    }
    return "";
  }
  return "";
}

CNodePtr DynamicObfuscator::password_mode_control(FuncGraphPtr func_graph) {
  ShapeVector y_shape{1, 1};
  tensor::TensorPtr y_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeInt32, y_shape);
  if (!has_build_appended_input) {
    MS_LOG(INFO) << "Build parameter y and y_append.";
    auto y = func_graph->add_parameter();
    y->set_name("y");
    y->set_abstract(y_tensor->ToAbstract());
    auto y_append = func_graph->add_parameter();
    y_append->set_name("y_append");
    y_append->set_abstract(y_tensor->ToAbstract());
    has_build_appended_input = true;
  }
  auto y = func_graph->GetParameterByName("y");
  auto y_append = func_graph->GetParameterByName("y_append");

  if (used_control_node_ == 0) {
    // make add function node
    mindspore::PrimitivePtr add_prim = mindspore::prim::kPrimAdd;
    add_prim->set_attr("is_load", MakeValue(true));
    mindspore::ValueNodePtr add_v_node = std::make_shared<mindspore::ValueNode>(add_prim);
    (void)func_graph->AddValueNode(add_v_node);
    CNodePtr add_c_node = func_graph->NewCNode({add_v_node, y, y_append});
    add_c_node->set_abstract(y_tensor->ToAbstract());
    // make equal function node
    ValueNodePtr equal_v_node = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimEqual);
    (void)func_graph->AddValueNode(equal_v_node);
    ValueNodePtr equal_compa_node = make_int_node(func_graph, obf_password_ + append_password_);
    CNodePtr equal_c_node = func_graph->NewCNode({equal_v_node, add_c_node, equal_compa_node});
    tensor::TensorPtr equal_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeBool, y_shape);
    equal_c_node->set_abstract(equal_tensor->ToAbstract());
    (void)func_graph->AddNode(equal_c_node);
    used_control_node_ += 1;
    switch_branch_ = true;
    return equal_c_node;
  }
  // make greater function node
  int comparison_int = rand();
  ValueNodePtr greater_v_node = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimGreater);
  (void)func_graph->AddValueNode(greater_v_node);
  ValueNodePtr greater_compa_node = make_int_node(func_graph, comparison_int);
  CNodePtr greater_c_node = func_graph->NewCNode({greater_v_node, y, greater_compa_node});
  tensor::TensorPtr greater_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeBool, y_shape);
  greater_c_node->set_abstract(greater_tensor->ToAbstract());
  (void)func_graph->AddNode(greater_c_node);
  used_control_node_ += 1;
  switch_branch_ = obf_password_ > comparison_int;
  return greater_c_node;
}

mindspore::CNodePtr AddStrideSliceNode(FuncGraphPtr func_graph, ShapeVector begin_vector, ShapeVector stride_vector,
                                       ShapeVector end_vector, int end_mask, int begin_mask,
                                       mindspore::CNodePtr prev_node) {
  mindspore::ValueNodePtr begin_v_node = build_tuple_value_node(begin_vector);
  mindspore::ValueNodePtr stride_v_node = build_tuple_value_node(stride_vector);
  mindspore::ValueNodePtr end_v_node = build_tuple_value_node(end_vector);
  (void)func_graph->AddValueNode(begin_v_node);
  (void)func_graph->AddValueNode(stride_v_node);
  (void)func_graph->AddValueNode(end_v_node);
  mindspore::PrimitivePtr slice_prim = mindspore::prim::kPrimStridedSlice;
  slice_prim->set_attr("is_load", MakeValue(true));
  slice_prim->set_attr("new_axis_mask", MakeValue(int64_t(0)));
  slice_prim->set_attr("shrink_axis_mask", MakeValue(int64_t(1)));
  slice_prim->set_attr("end_mask", MakeValue(int64_t(end_mask)));
  slice_prim->set_attr("begin_mask", MakeValue(int64_t(begin_mask)));
  slice_prim->set_attr("ellipsis_mask", MakeValue(int64_t(0)));
  mindspore::ValueNodePtr slice_v_node = std::make_shared<mindspore::ValueNode>(slice_prim);
  (void)func_graph->AddValueNode(slice_v_node);
  mindspore::CNodePtr slice_c_node =
    func_graph->NewCNode({slice_v_node, prev_node, begin_v_node, end_v_node, stride_v_node});
  return slice_c_node;
}

CNodePtr DynamicObfuscator::custom_op_mode_control(FuncGraphPtr func_graph, AnfNodePtr prev_node) {
  mindspore::PrimitivePtr reshape_prim = mindspore::prim::kPrimReshape;
  reshape_prim->set_attr("is_load", MakeValue(true));
  mindspore::ValueNodePtr reshape_v_node = std::make_shared<mindspore::ValueNode>(reshape_prim);
  (void)func_graph->AddValueNode(reshape_v_node);
  ShapeVector prev_node_shape = get_node_shape(prev_node);
  int shape_multiply = std::accumulate(prev_node_shape.begin(), prev_node_shape.end(), 1, std::multiplies<int>());
  MS_LOG(INFO) << "The shape_multiply is: " << shape_multiply;

  ShapeVector flat_shape{1, shape_multiply};
  mindspore::ValueNodePtr shape_v_node = std::make_shared<mindspore::ValueNode>(MakeValue(flat_shape));
  (void)func_graph->AddValueNode(shape_v_node);
  mindspore::CNodePtr reshape_c_node = func_graph->NewCNode({reshape_v_node, prev_node, shape_v_node});
  TypeId data_type = get_node_dtype(prev_node);
  auto reshape_abstract = std::make_shared<Tensor>(data_type, flat_shape)->ToAbstract();
  reshape_c_node->set_abstract(reshape_abstract);
  (void)func_graph->AddNode(reshape_c_node);

  // the first stride_slice x[0]
  ShapeVector begin_1{0, 0};
  ShapeVector stride_1{1, 1};
  mindspore::CNodePtr slice_c_node_1 =
    AddStrideSliceNode(func_graph, begin_1, stride_1, flat_shape, 2, 2, reshape_c_node);
  ShapeVector slice_1_shape{shape_multiply};
  slice_c_node_1->set_abstract(std::make_shared<Tensor>(data_type, slice_1_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_1);

  // the first stride_slice x[0][0]
  ShapeVector begin_2{0};
  ShapeVector end_2{1};
  ShapeVector stride_2{1};
  mindspore::CNodePtr slice_c_node_2 =
    AddStrideSliceNode(func_graph, begin_2, stride_2, stride_2, 0, 0, slice_c_node_1);
  ShapeVector slice_2_shape{1};
  slice_c_node_2->set_abstract(std::make_shared<Tensor>(data_type, slice_2_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_2);

  // the second stride_slice x[0][1]
  ShapeVector begin_3{1};
  ShapeVector end_3{1};
  ShapeVector stride_3{2};
  mindspore::CNodePtr slice_c_node_3 =
    AddStrideSliceNode(func_graph, begin_3, stride_3, stride_3, 0, 0, slice_c_node_1);
  ShapeVector slice_3_shape{1};
  slice_c_node_3->set_abstract(std::make_shared<Tensor>(data_type, slice_3_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_3);

  // add opaque predicate
  PrimitivePtr custom_prim = mindspore::prim::kPrimOpaquePredicate;
  custom_prim->set_attr("is_load", MakeValue(true));
  std::vector<ValuePtr> input_names_value;
  input_names_value.push_back(std::make_shared<StringImm>("x"));
  input_names_value.push_back(std::make_shared<StringImm>("y"));
  custom_prim->set_attr("input_names", std::make_shared<ValueList>(input_names_value));
  std::vector<ValuePtr> output_names_value;
  output_names_value.push_back(std::make_shared<StringImm>("output"));
  custom_prim->set_attr("output_names", std::make_shared<ValueList>(output_names_value));
  auto opaque_v_node = std::make_shared<mindspore::ValueNode>(custom_prim);
  (void)func_graph->AddValueNode(opaque_v_node);
  auto opaque_c_node = func_graph->NewCNode({opaque_v_node, slice_c_node_2, slice_c_node_3});
  ShapeVector y_shape{1, 1};
  auto bool_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeBool, y_shape);
  opaque_c_node->set_abstract(bool_tensor->ToAbstract());
  (void)func_graph->AddNode(opaque_c_node);
  return opaque_c_node;
}

CNodePtr DynamicObfuscator::get_control_node(FuncGraphPtr func_graph, AnfNodePtr prev_node) {
  if (obf_password_ != 0) {
    MS_LOG(INFO) << "Run password mode.";
    return password_mode_control(func_graph);
  }
  MS_LOG(INFO) << "Run customized function mode.";
  return custom_op_mode_control(func_graph, prev_node);
}

void DynamicObfuscator::replace_matmul_node(CNodePtr node, FuncGraphPtr func_graph, CNodePtr control_node) {
  std::vector<AnfNodePtr> node_inputs = node->cast<mindspore::CNodePtr>()->inputs();
  mindspore::ValueNodePtr matmul_v_node = node_inputs[0]->cast<mindspore::ValueNodePtr>();
  mindspore::AnfNodePtr input_1 = node_inputs[1];
  mindspore::AnfNodePtr input_2 = node_inputs[2];

  // construct branch 1
  mindspore::FuncGraphPtr fg_1 = std::make_shared<FuncGraph>();

  // input_x
  ParameterPtr branch_1_input_x = fg_1->add_parameter();
  branch_1_input_x->set_abstract(input_1->abstract());
  branch_1_input_x->set_name("branch_1_input_x");

  // input_y
  ParameterPtr branch_1_input_y = fg_1->add_parameter();
  branch_1_input_y->set_abstract(input_2->abstract());
  branch_1_input_y->set_name("branch_1_input_y");

  mindspore::CNodePtr matmul_c_node_1 = fg_1->NewCNode({matmul_v_node, branch_1_input_x, branch_1_input_y});
  matmul_c_node_1->set_abstract(node->cast<mindspore::CNodePtr>()->abstract());
  (void)fg_1->AddNode(matmul_c_node_1);

  // add return node
  mindspore::ValueNodePtr return_v_node_1 = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimReturn);
  (void)fg_1->AddValueNode(return_v_node_1);
  mindspore::CNodePtr branch_1_return = fg_1->NewCNode({return_v_node_1, matmul_c_node_1});
  (void)fg_1->AddNode(branch_1_return);
  fg_1->set_return(branch_1_return);
  fg_1->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);

  mindspore::ValueNodePtr partial_v_node_1 = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimPartial);
  (void)func_graph->AddValueNode(partial_v_node_1);
  mindspore::ValueNodePtr fg_1_node = std::make_shared<mindspore::ValueNode>(fg_1);
  fg_1_node->set_abstract(fg_1->ToAbstract());
  (void)func_graph->AddValueNode(fg_1_node);
  mindspore::CNodePtr partial_c_node_1 = func_graph->NewCNode({partial_v_node_1, fg_1_node, input_1, input_2});
  (void)func_graph->AddNode(partial_c_node_1);

  // construct branch 2
  mindspore::FuncGraphPtr fg_2 = std::make_shared<FuncGraph>();
  // add input_x
  ParameterPtr branch_2_input_x = fg_2->add_parameter();
  branch_2_input_x->set_abstract(input_1->abstract());
  branch_2_input_x->set_name("branch_2_input_x");
  // add input_y
  ParameterPtr branch_2_input_y = fg_2->add_parameter();
  branch_2_input_y->set_abstract(input_2->abstract());
  branch_2_input_y->set_name("branch_2_input_y");

  // add matmul CNode
  mindspore::CNodePtr matmul_c_node_2 = fg_2->NewCNode({matmul_v_node, branch_2_input_x, branch_2_input_y});
  matmul_c_node_2->set_abstract(node->cast<mindspore::CNodePtr>()->abstract());
  (void)fg_2->AddNode(matmul_c_node_2);

  // add return node
  mindspore::ValueNodePtr return_v_node_2 = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimReturn);
  (void)fg_2->AddValueNode(return_v_node_2);
  mindspore::CNodePtr branch_2_return = fg_2->NewCNode({return_v_node_2, matmul_c_node_2});
  (void)fg_2->AddNode(branch_2_return);
  fg_2->set_return(branch_2_return);
  fg_2->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);

  // add partial for branch_2
  ShapeVector matmul_2_shape = get_node_shape(input_2);
  TypeId type_id = get_node_dtype(input_2);
  tensor::TensorPtr matmul_2_weight = make_weight_tensor(type_id, matmul_2_shape);
  mindspore::ValueNodePtr matmul_weight_v_node = std::make_shared<mindspore::ValueNode>(matmul_2_weight);
  matmul_weight_v_node->set_abstract(matmul_2_weight->ToAbstract());
  (void)func_graph->AddValueNode(matmul_weight_v_node);

  mindspore::ValueNodePtr partial_v_node_2 = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimPartial);
  (void)func_graph->AddValueNode(partial_v_node_2);
  mindspore::ValueNodePtr fg_2_node = std::make_shared<mindspore::ValueNode>(fg_2);
  fg_2_node->set_abstract(fg_2->ToAbstract());
  (void)func_graph->AddValueNode(fg_2_node);
  mindspore::CNodePtr partial_c_node_2 =
    func_graph->NewCNode({partial_v_node_2, fg_2_node, input_1, matmul_weight_v_node});
  (void)func_graph->AddNode(partial_c_node_2);

  // add switch node
  mindspore::ValueNodePtr switch_v_node = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimSwitch);
  (void)func_graph->AddValueNode(switch_v_node);
  mindspore::CNodePtr switch_c_node;
  if (obf_password_ == 0) {
    int results_len = customized_func_results_.size();
    switch_branch_ = customized_func_results_[results_len - 1 - used_control_node_];
    used_control_node_ += 1;
  }
  if (switch_branch_) {
    switch_c_node = func_graph->NewCNode({switch_v_node, control_node, partial_c_node_1, partial_c_node_2});
  } else {
    switch_c_node = func_graph->NewCNode({switch_v_node, control_node, partial_c_node_2, partial_c_node_1});
  }
  func_graph->AddNode(switch_c_node);

  // add call node
  mindspore::CNodePtr call_cnode = func_graph->NewCNode({switch_c_node});
  func_graph->AddNode(call_cnode);
  // add fg_1 and fg_2 to func_graph
  auto mgr = mindspore::Manage(func_graph);
  mgr->AddFuncGraph(fg_1);
  mgr->AddFuncGraph(fg_2);
  mgr->Replace(node, call_cnode);
}
}  // namespace mindspore
