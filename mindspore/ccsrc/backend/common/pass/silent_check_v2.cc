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
#include "backend/common/pass/silent_check_v2.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <regex>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/param_info.h"
#include "ir/primal_attr.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/array_ops.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/framework_ops.h"
#include "ops/l2_normalize.h"
#include "ops/nn_ops.h"
#include "ops/other_ops.h"
#include "ops/sequence_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIndexOne = 1;
constexpr size_t kInputSizeTwo = 2;
constexpr char kScaleSense[] = "scale_sense";
constexpr char kNpuAsdEnable[] = "NPU_ASD_ENABLE";
constexpr char kParamSfdaPrefix[] = "silent_check_v2.sfda";
constexpr char kParamStepPrefix[] = "silent_check_v2.step";
constexpr char kNameSilentCheckV2[] = "SilentCheckV2";
constexpr int kMinStepDefault = 100;

std::string ltrim(const std::string &str) { return std::regex_replace(str, std::regex("^\\s+"), std::string("")); }

std::string rtrim(const std::string &str) { return std::regex_replace(str, std::regex("\\s+$"), std::string("")); }

std::string trim(const std::string &str) { return ltrim(rtrim(str)); }

std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    result.emplace_back(item);
  }

  return result;
}

// parse string in format "value0,value1" satisfying value0 > value0 to two float values
std::vector<float> parse_thresh(const std::string &value, float min_val) {
  std::vector<float> values;
  auto items = split(value, ',');
  if (items.size() != 2) {
    return values;
  }
  try {
    for (const auto &elem : items) {
      float val = std::stoll(trim(elem));
      if (val < min_val) {
        val = min_val;
      }
      values.push_back(val);
    }
  } catch (std::logic_error const &ex) {
    return {};
  }
  if (values.front() <= values.back()) {
    return {};
  }
  return values;
}

std::vector<float> parse_thresh(const std::string &env_var, const std::string &default_val, float min_val) {
  auto env_value = common::GetEnv(env_var);
  auto values = parse_thresh(env_value, min_val);
  if (!values.empty()) {
    return values;
  }

  if (!env_value.empty()) {
    MS_LOG(WARNING) << "Value of environment var " << env_var << " is invalid, use default value " << default_val
                    << " instead.";
  }

  values = parse_thresh(default_val, min_val);
  if (values.empty()) {
    MS_LOG(EXCEPTION) << "Default value of environment var " << env_var << " is invalid, of which value is "
                      << default_val;
  }
  return values;
}

int GetNpuAsdDetectValue() {
  auto var_val = common::GetEnv(kNpuAsdEnable);
  if (var_val.empty()) {
    return 0;
  }

  if (var_val.size() != 1 || var_val[0] < '0' || var_val[0] > '3') {
    MS_LOG(WARNING) << "Valid values of " << kNpuAsdEnable << " are 0, 1, 2 and 3, but got " << var_val << ".";
    return 0;
  }

  return var_val[0] - '0';
}

bool IsCommOperator(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return false;
  }
  auto prim = GetValuePtr<Primitive>(node);
  return common::AnfAlgo::IsCommunicationOp(prim->name()) && (prim->name() != kBarrierOpName);
}

bool GradCommOperatorUnvisited(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);

    // skip non-communication operators
    if (!IsCommOperator(node)) {
      return false;
    }
    return true;
  }
  return false;
}

// Get abstract of the default value in the given parameter.
AbstractBasePtr GetDefaultValueAbstract(const ParameterPtr &param) {
  auto value = param->default_param();
  MS_EXCEPTION_IF_NULL(value);
  auto value_abs = value->ToAbstract();
  MS_EXCEPTION_IF_NULL(value_abs);
  if (value_abs->isa<abstract::AbstractMapTensor>()) {
    // Return AbstractMapTensor for map parameter.
    return value_abs;
  }
  // Make an AbstractRefTensor for the tensor value.
  auto abs_tensor = value_abs->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(abs_tensor);
  auto ref_key = std::make_shared<RefKey>(param->name());
  return std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
}

AnfNodePtr CreateValueNode(const ValuePtr &value) {
  auto value_node = std::make_shared<ValueNode>(value);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(value->ToAbstract());
  return value_node;
}
}  // namespace

bool IsNpuAsdEnable() {
  auto ctx = MsContext::GetInstance();
  auto device_target = ctx->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return false;
  }
  if (ctx->ascend_soc_version() == kAscendVersion910) {
    return false;
  }
  return GetNpuAsdDetectValue() > 0;
}

const BaseRef SilentCheckV2::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(GradCommOperatorUnvisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

ParameterPtr CreateSfdaParam(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  static int param_sfda_index = 0;
  auto param_name = std::string(kParamSfdaPrefix) + std::to_string(param_sfda_index++);
  auto param_info = std::make_shared<ParamInfo>();
  param_info->set_requires_grad(false);
  param_info->set_name(param_name);

  // set initial sfda value to 0.0
  float sfda_init[] = {0.0, 0.0, 0.0};
  auto param_default_value =
    std::make_shared<tensor::Tensor>(kNumberTypeFloat32, ShapeVector{3}, sfda_init, sizeof(sfda_init));
  param_default_value->set_param_info(param_info);

  auto param = func_graph->add_parameter();
  param->set_name(param_name);
  param->set_default_param(param_default_value);
  param->set_abstract(GetDefaultValueAbstract(param));

  return param;
}

ParameterPtr CreateStepParam(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  static int param_sfda_index = 0;
  auto param_name = std::string(kParamStepPrefix) + std::to_string(param_sfda_index++);
  auto param_info = std::make_shared<ParamInfo>();
  param_info->set_requires_grad(false);
  param_info->set_name(param_name);

  // set initial step values to 0
  int64_t step_init[] = {0};
  auto param_default_value =
    std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{1}, step_init, sizeof(step_init));
  param_default_value->set_param_info(param_info);

  auto param = func_graph->add_parameter();
  param->set_name(param_name);
  param->set_default_param(param_default_value);
  param->set_abstract(GetDefaultValueAbstract(param));

  return param;
}

AnfNodePtr CreateNormForGE(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout) {
  std::vector<AnfNodePtr> square_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameSquare)), dout};
  auto square_node = func_graph->NewCNode(square_inputs);
  MS_EXCEPTION_IF_NULL(square_node);
  square_node->set_abstract(dout->abstract());
  square_node->set_scope(node->scope());

  auto reduce_axes = CreateValueNode(std::make_shared<ValueTuple>(std::vector<ValuePtr>{}));
  // set keep_dims and skip_mode to False
  auto false_node = CreateValueNode(std::make_shared<BoolImm>(false));
  std::vector<AnfNodePtr> reduce_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameReduceSum)), square_node,
                                           reduce_axes, false_node, false_node};
  auto reduce_node = func_graph->NewCNode(reduce_inputs);
  MS_EXCEPTION_IF_NULL(reduce_node);
  auto ret_abs = dout->abstract()->Clone();
  ret_abs->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{}));
  reduce_node->set_abstract(ret_abs);
  reduce_node->set_scope(node->scope());

  std::vector<AnfNodePtr> sqrt_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameSqrt)), reduce_node};
  auto sqrt_node = func_graph->NewCNode(sqrt_inputs);
  MS_EXCEPTION_IF_NULL(sqrt_node);
  sqrt_node->set_abstract(reduce_node->abstract());
  sqrt_node->set_scope(node->scope());

  return sqrt_node;
}

AnfNodePtr CreateNormForKBK(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout) {
  auto ord = CreateValueNode(std::make_shared<FP32Imm>(2.0));
  auto dims = CreateValueNode(std::make_shared<ValueTuple>(std::vector<ValuePtr>{}));
  auto keep_dims = CreateValueNode(std::make_shared<BoolImm>(false));
  std::vector<AnfNodePtr> norm_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameNorm)), dout, ord, dims,
                                         keep_dims};
  auto norm_node = func_graph->NewCNode(norm_inputs);
  MS_EXCEPTION_IF_NULL(norm_node);
  auto norm_abs = dout->abstract()->Clone();
  norm_abs->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{}));
  norm_node->set_abstract(norm_abs);
  norm_node->set_scope(node->scope());

  return norm_node;
}

AnfNodePtr GetGradValue(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout,
                        const ParameterPtr &loss_scale) {
  if (loss_scale == nullptr) {
    return dout;
  }

  auto umonad_node = NewValueNode(std::make_shared<UMonad>());
  umonad_node->set_abstract(std::make_shared<abstract::AbstractUMonad>());
  std::vector<AnfNodePtr> load_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimLoad->name())), loss_scale,
                                         umonad_node};
  auto load_node = func_graph->NewCNode(load_inputs);
  MS_EXCEPTION_IF_NULL(load_node);
  auto scale_param_abs = loss_scale->abstract()->cast<abstract::AbstractTensorPtr>();
  load_node->set_abstract(scale_param_abs->abstract::AbstractTensor::Clone());
  load_node->set_scope(node->scope());

  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameDiv)), dout, load_node};
  auto div_node = func_graph->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_abstract(dout->abstract());
  div_node->set_scope(node->scope());

  return div_node;
}

// replace print(i1, i2, U) with print(dummy_input, i1, i2, U) and set attributes of print
const AnfNodePtr SilentCheckV2::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // skip forward node in graph
  if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    return node;
  }

  if (cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << ", has attr forward_unique_id=" << std::boolalpha
                    << GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId))
                    << "inputs: " << cnode->DebugString();
  }

  // create SlientCheckV2 node
  auto check_prim = std::make_shared<Primitive>(kNameSilentCheckV2);
  check_prim->AddAttr("side_effect_mem", std::make_shared<BoolImm>(true));
  // input1: input_grad
  auto dout = GetGradValue(func_graph, node, cnode->input(kIndexOne), loss_scale_);
  // input0: val
  auto norm_node = MsContext::GetInstance()->GetJitLevel() == kAttrJitLevelO2
                     ? CreateNormForGE(func_graph, node, dout)
                     : CreateNormForKBK(func_graph, node, dout);
  // input2: sfda
  auto param_sfda = CreateSfdaParam(func_graph, node);
  // input3: step
  auto param_step = CreateStepParam(func_graph, node);
  // input4: cMinSteps
  auto min_steps = CreateValueNode(std::make_shared<Int64Imm>(kMinStepDefault));
  // input5: cThreshL1
  auto upper_thresh = parse_thresh("NPU_ASD_UPPER_THRESH", "1000000,10000", 3);
  auto thresh_l1 = CreateValueNode(std::make_shared<FP32Imm>(upper_thresh.front()));
  // input7: cThreshL2
  auto thresh_l2 = CreateValueNode(std::make_shared<FP32Imm>(upper_thresh.back()));
  // input6: cCoeffL1
  auto sigma_thresh = parse_thresh("NPU_ASD_SIGMA_THRESH", "100000,5000", 3);
  auto coeff_l1 = CreateValueNode(std::make_shared<FP32Imm>(sigma_thresh.front()));
  // input8: cCoeffL2
  auto coeff_l2 = CreateValueNode(std::make_shared<FP32Imm>(sigma_thresh.back()));
  // input9: npuAsdDetect
  auto npu_asd_detect = CreateValueNode(std::make_shared<Int64Imm>(GetNpuAsdDetectValue()));
  std::vector<AnfNodePtr> check_inputs = {NewValueNode(check_prim),
                                          norm_node,
                                          dout,
                                          param_sfda,
                                          param_step,
                                          min_steps,
                                          thresh_l1,
                                          coeff_l1,
                                          thresh_l2,
                                          coeff_l2,
                                          npu_asd_detect};
  auto check_node = func_graph->NewCNode(check_inputs);
  MS_EXCEPTION_IF_NULL(check_node);
  // output0: input_grad
  auto out_input_grad_abs = dout->abstract();
  // output1: sfda
  auto out_sfda_abs = param_sfda->abstract()->cast<abstract::AbstractTensorPtr>()->abstract::AbstractTensor::Clone();
  // output2: step
  auto out_step_abs = param_step->abstract()->cast<abstract::AbstractTensorPtr>()->abstract::AbstractTensor::Clone();
  // output3: result
  auto out_result_abs = std::make_shared<abstract::AbstractTensor>(kInt32, ShapeVector{});
  check_node->set_abstract(std::make_shared<abstract::AbstractTuple>(
    AbstractBasePtrList{out_input_grad_abs, out_sfda_abs, out_step_abs, out_result_abs}));
  check_node->set_scope(node->scope());

  // create Depend node
  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(std::make_shared<Primitive>(kDependOpName)),
                                           cnode->input(kIndexOne), check_node};
  auto depend_node = func_graph->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(dout->abstract());
  depend_node->set_scope(node->scope());

  // create new communication node to replace old node
  std::vector<AnfNodePtr> comm_inputs = cnode->inputs();
  comm_inputs[kIndexOne] = depend_node;
  auto new_comm_node = func_graph->NewCNode(comm_inputs);
  MS_EXCEPTION_IF_NULL(new_comm_node);
  new_comm_node->set_abstract(node->abstract());
  new_comm_node->set_scope(node->scope());
  new_comm_node->set_fullname_with_scope(node->fullname_with_scope());

  return new_comm_node;
}

void SilentCheckV2::GetLossScale() {
  MS_EXCEPTION_IF_NULL(root_);
  auto parameters = root_->parameters();
  for (const auto &param : parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    const auto &name = param_ptr->name();
    if (name == kScaleSense) {
      loss_scale_ = param_ptr;
    }
  }
}
}  // namespace opt
}  // namespace mindspore
