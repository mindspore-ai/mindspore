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
#include "utils/dynamic_obfuscation/dynamic_obfuscation.h"
#include <cstdlib>
#include <algorithm>
#include <map>
#include <memory>
#include <functional>
#include <random>
#include "utils/dynamic_obfuscation/registry_opaque_predicate.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/info.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace {
ParameterPtr AddObfuscatedParam(FuncGraphPtr func_graph) {
  auto params = func_graph->parameters();
  auto add_param = std::make_shared<Parameter>(func_graph);
  std::vector<AnfNodePtr> new_para_list(params.begin(), params.begin() + params.size() - func_graph->fv_param_count());
  (void)new_para_list.emplace_back(add_param);
  new_para_list.insert(new_para_list.cend(), params.begin() + params.size() - func_graph->fv_param_count(),
                       params.end());
  func_graph->set_parameters(new_para_list);
  return add_param;
}
}  // namespace
using Tensor = mindspore::tensor::Tensor;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

constexpr int keyExpandRate = 10;  // total node need for a switch graph
constexpr int kWeightIndex = 2;
constexpr int kSwitchInputsNum = 2;
constexpr int kNodeWithWeightInputsNum = 3;
constexpr auto keyConv2DOpName = "Conv2D-op";
constexpr auto keyReluOpName = "ReLU-op";
constexpr auto keySigmoidOpName = "Sigmoid-op";
constexpr auto keyMatMulOpName = "MatMul-op";

ShapeVector get_node_shape(const AnfNodePtr &input_node) {
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, get shape failed!";
    return {};
  }
  AbstractBasePtr input_abstract = input_node->abstract();
  if (input_abstract == nullptr) {
    MS_LOG(ERROR) << "The abstract of input_node is nullptr, get shape failed!";
    return {};
  }
  AbstractTensorPtr input_abstract_tensor = input_abstract->cast<mindspore::abstract::AbstractTensorPtr>();
  mindspore::abstract::ShapePtr shape_ptr = input_abstract_tensor->shape();
  if (shape_ptr == nullptr) {
    return {};
  }
  return shape_ptr->shape();
}

TypeId get_node_dtype(const AnfNodePtr &input_node) {
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, get dtype failed!";
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

std::vector<std::string> name_split(const std::string &node_name_, const std::string &split_sign) {
  std::string node_name = node_name_;
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

std::string get_node_name(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, get name failed!";
    return "";
  }
  std::string node_name = node->fullname_with_scope();
  std::vector<string> split_words = name_split(node_name, "/");
  if (split_words.empty()) {
    MS_LOG(WARNING) << "Input node name is empty.";
    return "";
  }
  std::string name = split_words[split_words.size() - 1];
  return name;
}

int get_op_num(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, get name failed!";
    return 0;
  }
  std::string node_name = node->fullname_with_scope();
  std::vector<string> split_words = name_split(node_name, "op");
  if (split_words.empty()) {
    MS_LOG(WARNING) << "Input node name is empty.";
    return 0;
  }
  std::string num = split_words[split_words.size() - 1];
  return std::stoi(num);
}

ParameterPtr get_node_param(FuncGraphPtr func_graph, const CNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Node is nullptr, get param failed!";
    return nullptr;
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "FuncGraph is nullptr, get param failed!";
    return nullptr;
  }
  std::string parameter_name = "";
  for (auto input : node->inputs()) {
    std::string op_name = get_node_name(input);
    int op_name_len = op_name.size();
    int load_len = 7;
    if ((op_name_len >= load_len) && (op_name.substr(0, load_len) == "Load-op")) {
      for (auto param : input->cast<mindspore::CNodePtr>()->inputs()) {
        if (param->fullname_with_scope().find("weight") != std::string::npos) {
          parameter_name = param->fullname_with_scope();
          break;
        }
      }
    }
  }
  for (auto param : func_graph->parameters()) {
    auto param_node = param->cast<mindspore::ParameterPtr>();
    if (param_node == nullptr) {
      MS_LOG(ERROR) << "Param node is nullptr.";
      return nullptr;
    }
    if (param->fullname_with_scope() == parameter_name) {
      return param_node;
    }
  }
  return nullptr;
}

ValueNodePtr build_tuple_value_node(std::vector<int64_t> values) {
  mindspore::ValueNodePtr v_node = std::make_shared<mindspore::ValueNode>(MakeValue(values));
  AbstractBasePtrList abs_list;
  std::transform(values.cbegin(), values.cend(), std::back_inserter(abs_list), [](const int64 &item) {
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

  // do subgraph fake-branch obfuscation
  (void)SubGraphFakeBranch(func_graph);

  if (subgraph_obf_num_ == 0) {
    MS_LOG(WARNING)
      << "The model has not been obfuscated, which means obf_password or customized_func is not need to set.";
  }
  return func_graph;
}

std::string DynamicObfuscator::ObfuscateOpType(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, get name failed!";
    return "";
  }
  if (node->isa<CNode>()) {
    MS_LOG(INFO) << "The node_name is: " << node->fullname_with_scope();
    std::string op_name = get_node_name(node);
    std::vector<std::string> target_op_list;
    target_op_list.insert(target_op_list.end(), single_input_target_op_.begin(), single_input_target_op_.end());
    target_op_list.insert(target_op_list.end(), single_input_with_weight_target_op_.begin(),
                          single_input_with_weight_target_op_.end());
    for (const auto &target_op_name : target_op_list) {
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

ObfCase DynamicObfuscator::ObfuscateOpCase(std::string obf_type) {
  if (obf_type.empty()) {
    MS_LOG(ERROR) << "Obf_type is empty string.";
    return ObfCase::NotObfNode;
  }
  auto name_equal = [&obf_type](const std::string &s) { return s == obf_type; };
  if (std::any_of(single_input_target_op_.begin(), single_input_target_op_.end(), name_equal)) {
    return ObfCase::OneInputNoWeightNode;
  } else if (std::any_of(single_input_with_weight_target_op_.begin(), single_input_with_weight_target_op_.end(),
                         name_equal)) {
    return ObfCase::OneInputWithWeightNode;
  } else {
    return ObfCase::NotObfNode;
  }
}

CNodePtr DynamicObfuscator::PasswordModeControl(FuncGraphPtr func_graph) {
  ShapeVector y_shape{1, 1};
  tensor::TensorPtr y_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeInt32, y_shape);
  if (!has_build_appended_input) {
    MS_LOG(INFO) << "Build parameter y_password and y_append.";
    auto y = AddObfuscatedParam(func_graph);
    y->set_name("y_password");
    y->set_abstract(y_tensor->ToAbstract());
    auto y_append = AddObfuscatedParam(func_graph);
    y_append->set_name("y_append");
    y_append->set_abstract(y_tensor->ToAbstract());
    has_build_appended_input = true;
  }
  auto y = func_graph->GetParameterByName("y_password");
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

mindspore::CNodePtr add_stride_slice_node(FuncGraphPtr func_graph, ShapeVector begin_vector, ShapeVector stride_vector,
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

CNodePtr DynamicObfuscator::CustomOpModeControl(FuncGraphPtr func_graph, const AnfNodePtr &prev_node) {
  mindspore::PrimitivePtr reshape_prim = mindspore::prim::kPrimReshape;
  reshape_prim->set_attr("is_load", MakeValue(true));
  mindspore::ValueNodePtr reshape_v_node = std::make_shared<mindspore::ValueNode>(reshape_prim);
  (void)func_graph->AddValueNode(reshape_v_node);
  ShapeVector prev_node_shape = get_node_shape(prev_node);
  int shape_multiply = std::accumulate(prev_node_shape.cbegin(), prev_node_shape.cend(), 1, std::multiplies<int>());
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
    add_stride_slice_node(func_graph, begin_1, stride_1, flat_shape, 2, 2, reshape_c_node);
  ShapeVector slice_1_shape{shape_multiply};
  slice_c_node_1->set_abstract(std::make_shared<Tensor>(data_type, slice_1_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_1);

  // the first stride_slice x[0][0]
  ShapeVector begin_2{0};
  ShapeVector end_2{1};
  ShapeVector stride_2{1};
  mindspore::CNodePtr slice_c_node_2 =
    add_stride_slice_node(func_graph, begin_2, stride_2, stride_2, 0, 0, slice_c_node_1);
  ShapeVector slice_2_shape{1};
  slice_c_node_2->set_abstract(std::make_shared<Tensor>(data_type, slice_2_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_2);

  // the second stride_slice x[0][1]
  ShapeVector begin_3{1};
  ShapeVector end_3{1};
  ShapeVector stride_3{2};
  mindspore::CNodePtr slice_c_node_3 =
    add_stride_slice_node(func_graph, begin_3, stride_3, stride_3, 0, 0, slice_c_node_1);
  ShapeVector slice_3_shape{1};
  slice_c_node_3->set_abstract(std::make_shared<Tensor>(data_type, slice_3_shape)->ToAbstract());
  (void)func_graph->AddNode(slice_c_node_3);

  // add opaque predicate
  PrimitivePtr custom_prim = mindspore::prim::kPrimOpaquePredicate;
  custom_prim->set_attr("is_load", MakeValue(true));
  std::vector<ValuePtr> input_names_value;
  input_names_value.push_back(std::make_shared<StringImm>("x"));
  input_names_value.push_back(std::make_shared<StringImm>("y"));
  custom_prim->set_attr(mindspore::kAttrInputNames, std::make_shared<ValueList>(input_names_value));
  std::vector<ValuePtr> output_names_value;
  output_names_value.push_back(std::make_shared<StringImm>("output"));
  custom_prim->set_attr(mindspore::kAttrOutputNames, std::make_shared<ValueList>(output_names_value));
  auto opaque_v_node = std::make_shared<mindspore::ValueNode>(custom_prim);
  (void)func_graph->AddValueNode(opaque_v_node);
  auto opaque_c_node = func_graph->NewCNode({opaque_v_node, slice_c_node_2, slice_c_node_3});
  ShapeVector y_shape{1, 1};
  auto bool_tensor = std::make_shared<Tensor>(mindspore::kNumberTypeBool, y_shape);
  opaque_c_node->set_abstract(bool_tensor->ToAbstract());
  (void)func_graph->AddNode(opaque_c_node);
  return opaque_c_node;
}

CNodePtr DynamicObfuscator::GetControlNode(const FuncGraphPtr &func_graph, const AnfNodePtr &prev_node) {
  if (obf_password_ != 0) {
    MS_LOG(INFO) << "Run password mode.";
    return PasswordModeControl(func_graph);
  }
  MS_LOG(INFO) << "Run customized function mode.";
  if (prev_node != nullptr && prev_node->abstract() != nullptr) {
    return CustomOpModeControl(func_graph, prev_node);
  }
  return nullptr;
}

mindspore::PrimitivePtr DynamicObfuscator::get_random_prim(const std::string &obf_type,
                                                           const mindspore::CNodePtr &node) {
  std::vector<string> split_words = name_split(obf_type, "-");
  if (split_words.empty()) {
    MS_LOG(WARNING) << "obf_type is empty.";
    return nullptr;
  }
  std::string prim_name_ori = split_words[0];
  mindspore::PrimitivePtr poolptr = nullptr;
  if (prim_name_ori == kMaxPoolOpName || prim_name_ori == kAvgPoolOpName) {
    if (prim_name_ori == kMaxPoolOpName) {
      poolptr = std::make_shared<Primitive>("AvgPool");
    } else {
      poolptr = std::make_shared<Primitive>("MaxPool");
    }
    auto primitive = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(primitive);
    MS_EXCEPTION_IF_NULL(primitive->GetAttr("input_names"));
    MS_EXCEPTION_IF_NULL(primitive->GetAttr("output_names"));
    MS_EXCEPTION_IF_NULL(primitive->GetAttr("format"));
    MS_EXCEPTION_IF_NULL(primitive->GetAttr("kernel_size"));
    MS_EXCEPTION_IF_NULL(primitive->GetAttr("strides"));
    poolptr->set_attr("input_names", primitive->GetAttr("input_names"));
    poolptr->set_attr("output_names", primitive->GetAttr("output_names"));
    poolptr->set_attr("format", primitive->GetAttr("format"));
    poolptr->set_attr("pad_mode", primitive->GetAttr("pad_mode"));
    poolptr->set_attr("kernel_size", primitive->GetAttr("kernel_size"));
    poolptr->set_attr("strides", primitive->GetAttr("strides"));
    return poolptr;
  }
  mindspore::PrimitivePtr prim_node = one_input_prim_[0];
  do {
    int random = rand() % one_input_prim_.size();
    prim_node = one_input_prim_[random];
  } while (prim_name_ori == prim_node->ToString());
  return prim_node;
}

void DynamicObfuscator::UpdateDict(const AnfNodePtr &node, const bool isParent) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, update dict failed.";
    return;
  }
  MS_LOG(INFO) << "Update: " << node->fullname_with_scope() << " to dict.";
  if (isParent) {
    parent_names_.push(node->fullname_with_scope());
  } else {
    node_names_.push(node->fullname_with_scope());
    subgraph_obf_num_++;
  }
  node_dict_[node->fullname_with_scope()] = node->cast<mindspore::AnfNodePtr>();
  if (node_dict_[node->fullname_with_scope()] == nullptr) {
    MS_LOG(ERROR) << "Update node " << node_dict_[node->fullname_with_scope()] << " failed.";
  }
}

void DynamicObfuscator::CheckDuplicatedParent(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, check parent failed.";
    return;
  }
  if (node_dict_.find(node->fullname_with_scope()) != node_dict_.cend()) {
    while (node_names_.top() != "-") {
      node_dict_.erase(node_names_.top());
      node_names_.pop();
      subgraph_obf_num_--;
    }
  } else {
    node_names_.push("-");
    UpdateDict(node, true);
    if (obf_password_ == 0) {
      bool customized_func_result = mindspore::kernel::CustomizedOpaquePredicate::GetInstance().run_function(
        static_cast<float>(1), static_cast<float>(1));
      customized_func_results_.push_back(customized_func_result);
    }
  }
}

bool DynamicObfuscator::IsTarget(std::string &cnode_name) {
  std::vector<string> split_words = name_split(cnode_name, "/");
  if (split_words.empty()) {
    MS_LOG(WARNING) << "CNode name is empty.";
    return false;
  }
  std::string op_name = split_words[split_words.size() - 1];
  std::vector<std::string> target_op_list;
  target_op_list.insert(target_op_list.end(), single_input_target_op_.begin(), single_input_target_op_.end());
  target_op_list.insert(target_op_list.end(), single_input_with_weight_target_op_.begin(),
                        single_input_with_weight_target_op_.end());
  for (std::string target_op_name : target_op_list) {
    int op_name_len = op_name.size();
    int target_name_len = target_op_name.size();
    if ((op_name_len >= target_name_len) && (op_name.substr(0, target_name_len) == target_op_name)) {
      return true;
    }
  }
  return false;
}

mindspore::CNodePtr DynamicObfuscator::CheckInputNodes(const mindspore::CNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr, check input failed.";
    return nullptr;
  }
  auto node_inputs = node->inputs();
  for (auto input_node : node_inputs) {
    std::string cnode_name = get_node_name(input_node);
    if (IsTarget(cnode_name)) {
      return input_node->cast<mindspore::CNodePtr>();
    }
  }
  return nullptr;
}

mindspore::CNodePtr DynamicObfuscator::BuildOneInputNoWeightNode(const FuncGraphPtr &fg,
                                                                 const mindspore::AnfNodePtr &input_node,
                                                                 const mindspore::PrimitivePtr prim_node) {
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Build Node failed: input node is nullptr.";
    return nullptr;
  }
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Build Node failed: FuncGraph is nullptr.";
    return nullptr;
  }
  std::vector<ValuePtr> input_names_value;
  input_names_value.emplace_back(std::make_shared<StringImm>("x"));
  prim_node->set_attr("is_load", MakeValue(true));
  prim_node->set_attr(mindspore::kAttrInputNames, std::make_shared<ValueList>(input_names_value));
  mindspore::ValueNodePtr v_node = std::make_shared<mindspore::ValueNode>(prim_node);
  (void)fg->AddValueNode(v_node);
  mindspore::CNodePtr c_node = fg->NewCNode({v_node, input_node});
  if (c_node == nullptr) {
    MS_LOG(ERROR) << "Build node failed: cnode is nullptr.";
    return nullptr;
  }
  ShapeVector x_shape = get_node_shape(input_node);
  TypeId type_id = get_node_dtype(input_node);
  auto node_abstract = std::make_shared<Tensor>(type_id, x_shape)->ToAbstract();
  if (node_abstract == nullptr) {
    MS_LOG(ERROR) << "Build node failed: node abstract is nullptr.";
    return nullptr;
  }
  c_node->set_abstract(node_abstract);
  (void)fg->AddNode(c_node);
  return c_node;
}

mindspore::CNodePtr DynamicObfuscator::BuildOneInputWithWeightNode(const FuncGraphPtr &fg,
                                                                   const mindspore::AnfNodePtr &input_node,
                                                                   const mindspore::CNodePtr &node,
                                                                   const mindspore::AnfNodePtr &weights) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Build one input with weight node failed: node is nullptr.";
    return nullptr;
  }
  std::string node_name = get_node_name(node);
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: input node is nullptr.";
    return nullptr;
  }
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: FuncGraph is nullptr.";
    return nullptr;
  }
  if (weights == nullptr) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: weights is nullptr.";
    return nullptr;
  }
  std::vector<AnfNodePtr> node_inputs = node->inputs();
  if (node_inputs.size() < 1) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: inputs size is 0";
    return nullptr;
  }
  mindspore::ValueNodePtr v_node = node_inputs[0]->cast<mindspore::ValueNodePtr>();
  (void)fg->AddValueNode(v_node);

  mindspore::CNodePtr c_node = fg->NewCNode({v_node, input_node, weights});
  if (c_node == nullptr) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: cnode is nullptr.";
    return nullptr;
  }
  ShapeVector x_shape = get_node_shape(node);
  TypeId type_id = get_node_dtype(node);
  auto node_abstract = std::make_shared<Tensor>(type_id, x_shape)->ToAbstract();
  if (node_abstract == nullptr) {
    MS_LOG(ERROR) << "Build " << node_name << " failed: abstract is nullptr.";
    return nullptr;
  }
  c_node->set_abstract(node_abstract);
  (void)fg->AddNode(c_node);
  return c_node;
}

FuncGraphPtr DynamicObfuscator::CloneSubGraph(const FuncGraphPtr &fg, const std::vector<mindspore::CNodePtr> &node_arr,
                                              const mindspore::AnfNodePtr &parent_node) {
  MS_LOG(INFO) << "Building Clone Graph ";
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Build clone graph failed: FuncGraph is nullptr.";
  }
  mindspore::FuncGraphPtr fg_clone = std::make_shared<FuncGraph>();
  ShapeVector x_shape = get_node_shape(parent_node);
  TypeId x_type_id = get_node_dtype(parent_node);
  MS_LOG(INFO) << "Get Shape Input X";

  mindspore::ParameterPtr input_x = fg_clone->add_parameter();
  if (input_x == nullptr) {
    MS_LOG(ERROR) << "Build clone graph failed: input_x is nullptr.";
    return nullptr;
  }
  input_x->set_name("input_x_clone");
  tensor::TensorPtr input_x_tensor = std::make_shared<Tensor>(x_type_id, x_shape);
  input_x->set_abstract(input_x_tensor->ToAbstract());
  mindspore::AnfNodePtr last_node = input_x;
  for (auto node : node_arr) {
    std::string obf_type = ObfuscateOpType(node);
    mindspore::ObfCase obf_case = ObfuscateOpCase(obf_type);
    switch (static_cast<mindspore::ObfCase>(obf_case)) {
      case ObfCase::OneInputNoWeightNode: {
        mindspore::PrimitivePtr prim_node = GetCNodePrimitive(node);
        last_node = BuildOneInputNoWeightNode(fg_clone, last_node, prim_node);
        if (last_node == nullptr) {
          MS_LOG(ERROR) << "Last node after build is nullptr.";
          return nullptr;
        }
        break;
      }
      case ObfCase::OneInputWithWeightNode: {
        mindspore::ParameterPtr weight_param = fg_clone->add_parameter();
        if (weight_param == nullptr) {
          MS_LOG(ERROR) << "Build OneInputWithWeightNode failed: weights is nullptr.";
          return nullptr;
        }
        weight_param->set_name("OneInputWithWeightNode_clone");
        last_node = BuildOneInputWithWeightNode(fg_clone, last_node, node, weight_param);
        if (last_node == nullptr) {
          MS_LOG(ERROR) << "Last node after build is nullptr.";
          return nullptr;
        }
        break;
      }
      case ObfCase::NotObfNode: {
        MS_LOG(ERROR) << "The current node does not belong to target nodes.";
      }
      default:
        return nullptr;
    }
  }

  mindspore::ValueNodePtr return_v = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimReturn);
  (void)fg_clone->AddValueNode(return_v);
  mindspore::CNodePtr return_c_node = fg_clone->NewCNode({return_v, last_node});
  if (return_c_node == nullptr) {
    MS_LOG(ERROR) << "Build return failed: return cnode is nullptr.";
    return nullptr;
  }
  ShapeVector return_shape = get_node_shape(last_node->cast<mindspore::CNodePtr>());
  TypeId type_id = get_node_dtype(last_node->cast<mindspore::CNodePtr>());
  auto return_abstract = std::make_shared<Tensor>(type_id, return_shape)->ToAbstract();
  if (return_abstract == nullptr) {
    MS_LOG(ERROR) << "Build return failed: return abstract is nullptr.";
    return nullptr;
  }
  return_c_node->set_abstract(return_abstract);
  (void)fg_clone->AddNode(return_c_node);
  fg_clone->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  fg_clone->set_return(return_c_node);
  return fg_clone;
}

FuncGraphPtr DynamicObfuscator::BuildFakeGraph(const FuncGraphPtr &fg, const std::vector<mindspore::CNodePtr> &node_arr,
                                               const mindspore::AnfNodePtr &parent_node) {
  MS_LOG(INFO) << "Building Fake Graph ";
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Build fake graph failed: FuncGraph is nullptr.";
    return nullptr;
  }
  mindspore::FuncGraphPtr fg_fake = std::make_shared<FuncGraph>();

  ShapeVector x_shape = get_node_shape(parent_node);
  TypeId x_type_id = get_node_dtype(parent_node);
  mindspore::ParameterPtr input_x = fg_fake->add_parameter();
  if (input_x == nullptr) {
    MS_LOG(ERROR) << "Build fake graph failed: input_x is nullptr.";
    return nullptr;
  }
  input_x->set_name("input_x_fake");
  tensor::TensorPtr input_x_tensor = std::make_shared<Tensor>(x_type_id, x_shape);
  input_x->set_abstract(input_x_tensor->ToAbstract());
  mindspore::AnfNodePtr last_node = input_x;
  for (auto node : node_arr) {
    std::string obf_type = ObfuscateOpType(node);
    mindspore::ObfCase obf_case = ObfuscateOpCase(obf_type);
    switch (static_cast<mindspore::ObfCase>(obf_case)) {
      case ObfCase::OneInputNoWeightNode: {
        mindspore::PrimitivePtr prim_node = get_random_prim(obf_type, node);
        last_node = BuildOneInputNoWeightNode(fg_fake, last_node, prim_node);
        if (last_node == nullptr) {
          MS_LOG(ERROR) << "Last node after build is nullptr.";
          return nullptr;
        }
        break;
      }
      case ObfCase::OneInputWithWeightNode: {
        mindspore::ParameterPtr weight_param = fg_fake->add_parameter();
        if (weight_param == nullptr) {
          MS_LOG(ERROR) << "Build OneInputWithWeightNode failed: weights is nullptr.";
          return nullptr;
        }
        weight_param->set_name("OneInputWithWeightNode_fake");
        mindspore::AnfNodePtr ori_vnode = node->cast<mindspore::CNodePtr>()->inputs()[2];
        TypeId type_id = get_node_dtype(ori_vnode);
        ShapeVector shape = get_node_shape(ori_vnode);
        tensor::TensorPtr weight_tensor = make_weight_tensor(type_id, shape);
        mindspore::ValueNodePtr weight_vnode = std::make_shared<mindspore::ValueNode>(weight_tensor);
        weight_vnode->set_abstract(weight_tensor->ToAbstract());
        if (weight_vnode == nullptr) {
          MS_LOG(ERROR) << "Build OneInputWithWeightNode failed: value node is nullptr.";
          return nullptr;
        }
        (void)fg_fake->AddValueNode(weight_vnode);
        last_node = BuildOneInputWithWeightNode(fg_fake, last_node, node, weight_vnode);
        if (last_node == nullptr) {
          MS_LOG(ERROR) << "Last node after build is nullptr.";
          return nullptr;
        }
        break;
      }
      case ObfCase::NotObfNode: {
        MS_LOG(ERROR) << "The current node is not obf-target";
      }
      default:
        return nullptr;
    }
  }

  mindspore::ValueNodePtr return_v = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimReturn);
  (void)fg_fake->AddValueNode(return_v);
  mindspore::CNodePtr return_c_node = fg_fake->NewCNode({return_v, last_node});
  if (return_c_node == nullptr) {
    MS_LOG(ERROR) << "Build return failed: return cnode is nullptr.";
    return nullptr;
  }
  ShapeVector return_shape = get_node_shape(last_node->cast<mindspore::CNodePtr>());
  TypeId type_id = get_node_dtype(last_node->cast<mindspore::CNodePtr>());
  auto return_abstract = std::make_shared<Tensor>(type_id, return_shape)->ToAbstract();
  if (return_abstract == nullptr) {
    MS_LOG(ERROR) << "Build return failed: return abstract is nullptr.";
    return nullptr;
  }
  return_c_node->set_abstract(return_abstract);
  (void)fg_fake->AddNode(return_c_node);
  fg_fake->set_return(return_c_node);
  fg_fake->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  return fg_fake;
}

mindspore::CNodePtr DynamicObfuscator::AddPartialBranch(FuncGraphPtr fg, FuncGraphPtr fg_sub,
                                                        const std::vector<mindspore::CNodePtr> &nodes) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Add subgraph failed: fg is null.";
    return nullptr;
  }
  if (fg_sub == nullptr) {
    MS_LOG(ERROR) << "Add subgraph failed: fg_sub is null.";
    return nullptr;
  }
  if (nodes.size() == 0) {
    MS_LOG(ERROR) << "Add subgraph failed: input nodes size is 0.";
    return nullptr;
  }

  mindspore::ValueNodePtr switch_partial = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimPartial);
  (void)fg->AddValueNode(switch_partial);
  mindspore::ValueNodePtr fg_subgraph_node = std::make_shared<mindspore::ValueNode>(fg_sub);
  fg_subgraph_node->set_abstract(fg_sub->ToAbstract());
  (void)fg->AddValueNode(fg_subgraph_node);
  std::vector<mindspore::AnfNodePtr> subgraph_inputs = {switch_partial, fg_subgraph_node};
  if (nodes[0]->inputs().size() < kSwitchInputsNum) {
    MS_LOG(ERROR) << "Add subgraph failed: the input number of node[0] is smaller than " << kSwitchInputsNum;
    return nullptr;
  }
  subgraph_inputs.push_back(nodes[0]->inputs()[1]);
  for (unsigned i = 0; i < nodes.size(); i++) {
    std::string obf_type = ObfuscateOpType(nodes[i]);
    if ((obf_type == keyConv2DOpName || obf_type == keyMatMulOpName) &&
        nodes[i]->inputs().size() >= kNodeWithWeightInputsNum) {
      subgraph_inputs.push_back(nodes[i]->inputs()[kWeightIndex]);
    }
  }
  mindspore::CNodePtr switch_partial_c = fg->NewCNode(subgraph_inputs);
  if (switch_partial_c == nullptr) {
    MS_LOG(ERROR) << "Add subgraph failed: switch partial is null.";
    return nullptr;
  }
  switch_partial_c->set_abstract(fg_sub->ToAbstract());
  (void)fg->AddNode(switch_partial_c);
  return switch_partial_c;
}

void DynamicObfuscator::AddSwitchNode(FuncGraphPtr fg) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Build switch failed: FuncGraph is nullptr.";
    return;
  }
  int switch_num_ = 0;
  while (!parent_names_.empty()) {
    auto mgr = mindspore::Manage(fg);
    if (mgr == nullptr) {
      MS_LOG(ERROR) << "FuncGraph manager is nullptr.";
    }
    std::vector<mindspore::CNodePtr> nodes;
    mindspore::AnfNodePtr last_node = nullptr;
    mindspore::CNodePtr child_node = nullptr;
    while (node_names_.top() != "-") {
      MS_LOG(INFO) << "Processing sub_graph node: " << node_names_.top();
      last_node = node_dict_[node_names_.top()];
      nodes.push_back(last_node->cast<mindspore::CNodePtr>());
      node_names_.pop();  // pop '-'
    }
    node_names_.pop();
    if (mgr->node_users().find(last_node) != mgr->node_users().cend()) {
      auto users = mgr->node_users()[last_node];
      child_node = users.cbegin()->first->cast<mindspore::CNodePtr>();
    } else {
      MS_LOG(ERROR) << "Child Node is nullptr.";
    }
    mindspore::AnfNodePtr parent_node = node_dict_[parent_names_.top()];
    parent_names_.pop();

    mindspore::FuncGraphPtr fg_subgraph_clone = CloneSubGraph(fg, nodes, parent_node);
    mindspore::FuncGraphPtr fg_subgraph_fake = BuildFakeGraph(fg, nodes, parent_node);

    mgr->AddFuncGraph(fg_subgraph_clone);
    mgr->AddFuncGraph(fg_subgraph_fake);

    mindspore::CNodePtr switch_partial_clone_c = AddPartialBranch(fg, fg_subgraph_clone, nodes);
    mindspore::CNodePtr switch_partial_fake_c = AddPartialBranch(fg, fg_subgraph_fake, nodes);

    CNodePtr control_node = GetControlNode(fg, parent_node);
    if (control_node == nullptr) {
      continue;
    }

    mindspore::ValueNodePtr switch_v_node = std::make_shared<mindspore::ValueNode>(mindspore::prim::kPrimSwitch);
    (void)fg->AddValueNode(switch_v_node);
    mindspore::CNodePtr switch_c_node;
    if (obf_password_ == 0) {
      if (static_cast<int>(customized_func_results_.size()) <= used_control_node_) {
        MS_LOG(ERROR) << "customized_func_results_ size is smaller than used_control_node_.";
      }
      switch_branch_ = customized_func_results_[used_control_node_];
      used_control_node_ += 1;
    }
    if (switch_branch_) {
      switch_c_node = fg->NewCNode({switch_v_node, control_node, switch_partial_clone_c, switch_partial_fake_c});
    } else {
      switch_c_node = fg->NewCNode({switch_v_node, control_node, switch_partial_fake_c, switch_partial_clone_c});
    }
    switch_c_node->set_abstract(fg_subgraph_clone->ToAbstract());
    fg->AddNode(switch_c_node);

    mindspore::CNodePtr call_cnode = fg->NewCNode({switch_c_node});
    fg->AddNode(call_cnode);

    unsigned i = 0;
    for (auto input : child_node->inputs()) {
      if (input->fullname_with_scope() == last_node->fullname_with_scope()) {
        child_node->set_input(i, call_cnode);
        break;
      }
      i++;
    }
    switch_num_++;
  }
  MS_LOG(WARNING) << switch_num_ << " switch nodes have been added.";
  used_control_node_ = 0;
}

void DynamicObfuscator::SubGraphFakeBranch(FuncGraphPtr func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Build fake sub-graph failed: FuncGraph is nullptr.";
  }
  node_names_.push("-");
  auto mgr = mindspore::Manage(func_graph);
  if (mgr == nullptr) {
    MS_LOG(INFO) << "Manager is null node!";
  }
  auto all_nodes = mgr->all_nodes();
  int node_nums = all_nodes.size();
  int obfuscate_target_num = std::ceil(node_nums * obf_ratio_ / keyExpandRate);
  int op_num = node_nums;  // Initialize op_num to the maximum node number
  std::vector<mindspore::AnfNodePtr> sorted_nodes;
  for (auto node : all_nodes) {
    MS_LOG(INFO) << "The last node name is: " << node->fullname_with_scope();
    sorted_nodes = TopoSort(node);  // the node number in front of sorted nodes is the smallest
    break;
  }
  std::reverse(sorted_nodes.begin(), sorted_nodes.end());
  for (auto node : sorted_nodes) {
    if (node == nullptr) {
      MS_LOG(INFO) << "Find null node!" << std::endl;
    }
    if (node->isa<CNode>()) {
      std::string cnode_name = get_node_name(node);
      int cur_op_num = get_op_num(node);
      float dropout_rate = 0.1;
      int dropout_rand = rand() % static_cast<int>(1.0 / dropout_rate);
      if (IsTarget(cnode_name) && cur_op_num < op_num && dropout_rand != 0 &&
          (node_dict_.find(node->fullname_with_scope()) == node_dict_.cend())) {
        UpdateDict(node, false);
        op_num = cur_op_num;
        bool stop_traverse = false;
        mindspore::CNodePtr curr_cnode = node->cast<mindspore::CNodePtr>();
        while (!stop_traverse) {
          mindspore::CNodePtr valid_input = CheckInputNodes(curr_cnode);
          dropout_rand = rand() % static_cast<int>(1.0 / dropout_rate);
          if (valid_input && dropout_rand != 0 &&
              (node_dict_.find(valid_input->fullname_with_scope()) == node_dict_.cend())) {
            UpdateDict(valid_input, false);
            op_num = get_op_num(valid_input);
            curr_cnode = valid_input->cast<mindspore::CNodePtr>();
          } else {
            stop_traverse = true;
            if (curr_cnode->inputs().size() > 1) {
              CheckDuplicatedParent(curr_cnode->inputs()[1]);
            }
          }
        }
      }
    }
    if (subgraph_obf_num_ >= obfuscate_target_num) {
      break;
    }
  }
  node_names_.pop();
  if (obf_password_ == 0) {
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().init_calling_count();
  }
  (void)AddSwitchNode(func_graph);
  MS_LOG(WARNING) << subgraph_obf_num_ << " nodes have been obfuscated.";
}
}  // namespace mindspore
