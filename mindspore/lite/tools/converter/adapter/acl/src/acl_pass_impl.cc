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

#include "tools/converter/adapter/acl/src/acl_pass_impl.h"
#include <set>
#include <map>
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/spatial_node_adapter.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/common/string_util.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/src/acl_model_process.h"
#include "include/registry/pass_registry.h"
#include "ops/custom.h"
#include "ops/tuple_get_item.h"
#include "base/core_ops.h"
#include "cxx_api/model/acl/model_converter.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace opt {
static const std::set<std::string> kDevice = {"Ascend310", "Ascend710"};
static const std::set<std::string> kAdjustCnodeName = {"Resize", "Conv2dTransposeFusion", "Concat"};
static const std::map<int64_t, std::string> kEnumFormatToStrMap = {{Format::NCHW, "NCHW"}, {Format::NHWC, "NHWC"}};
namespace {
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kOutputNames = "outputs_names";
constexpr auto kCustomPrimTypeACL = "ACL";
constexpr auto kCustomNodeName = "custom_0";
constexpr auto kNCHWFormat = "NCHW";
constexpr auto kToNHWCFormatPass = "ToNHWCFormat";
constexpr auto kToNCHWFormatPass = "ToNCHWFormat";
constexpr auto kInferShapePass = "InferShapePass";
constexpr auto kConstFoldPass = "ConstFoldPass";
constexpr auto kRemoveRedundantOpPass = "RemoveRedundantOpPass";
constexpr auto kDelRedundantTranspose = "DeleteRedundantTranspose";
constexpr size_t kDependInputNum = 3;
constexpr size_t kDependFirstInputIdx = 1;
constexpr size_t kTupleGetItemFirstInputIdx = 1;

STATUS PreProcForMindIr(const FuncGraphPtr &func_graph) { return lite::RET_OK; }

STATUS PreProcForTF(const FuncGraphPtr &func_graph) {
  if (!lite::RunOptimizerPass(func_graph, {kInferShapePass})) {
    MS_LOG(ERROR) << "Infer shape pass failed.";
    return lite::RET_ERROR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    CHECK_NULL_RETURN(node);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    CHECK_NULL_RETURN(cnode);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(prim);
    if (prim->GetAttr(ops::kFormat) != nullptr) {
      auto node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
      if (kEnumFormatToStrMap.find(node_format) != kEnumFormatToStrMap.end()) {
        std::string format = kEnumFormatToStrMap.at(node_format);
        prim->AddAttr("io_format", MakeValue(format));
      }
    }
  }
  return lite::RET_OK;
}

STATUS PreProcForCaffe(const FuncGraphPtr &func_graph) {
  if (!lite::RunOptimizerPass(func_graph, {kInferShapePass, kToNCHWFormatPass, kDelRedundantTranspose})) {
    MS_LOG(ERROR) << "To nchw format failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS PreProcForOnnx(const FuncGraphPtr &func_graph) {
  if (!lite::RunOptimizerPass(func_graph, {kInferShapePass, kToNCHWFormatPass, kDelRedundantTranspose})) {
    MS_LOG(ERROR) << "To nchw format failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace

AclPassImpl::AclPassImpl(const converter::Flags &config)
    : device_type_(config.device),
      fmk_type_(config.fmk),
      user_options_cfg_(std::move(config.aclModelOptionCfgParam)),
      om_parameter_(nullptr),
      custom_node_(nullptr) {}

bool AclPassImpl::IsDeviceAscend() {
  if (kDevice.find(device_type_) != kDevice.end()) {
    return true;
  }
  return false;
}

bool AclPassImpl::IsDynamicInput() {
  return !user_options_cfg_.dynamic_image_size.empty() || !user_options_cfg_.dynamic_batch_size.empty();
}

STATUS AclPassImpl::CommonPass(const FuncGraphPtr &func_graph) {
  if (!lite::RunOptimizerPass(func_graph, {kRemoveRedundantOpPass})) {
    MS_LOG(ERROR) << "Remove redundant op pass failed.";
    return lite::RET_ERROR;
  }
  if (IsDynamicInput()) {
    MS_LOG(INFO) << "Dynamic input no need to run const fold pass.";
    return lite::RET_OK;
  }
  if (!lite::RunOptimizerPass(func_graph, {kConstFoldPass})) {
    MS_LOG(ERROR) << "Const fold pass failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::PreProcGraph(const FuncGraphPtr &func_graph) {
  if (CommonPass(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Common pass failed.";
    return lite::RET_ERROR;
  }
  std::map<converter::FmkType, std::function<STATUS(const FuncGraphPtr &)>> fmk_proc_func = {
    {converter::kFmkTypeMs, PreProcForMindIr},   {converter::kFmkTypeTf, PreProcForTF},
    {converter::kFmkTypeCaffe, PreProcForCaffe}, {converter::kFmkTypeOnnx, PreProcForOnnx},
    {converter::kFmkTypeTflite, PreProcForTF},
  };
  if (fmk_proc_func.find(fmk_type_) != fmk_proc_func.end()) {
    auto func = fmk_proc_func.at(fmk_type_);
    if (func(func_graph) != lite::RET_OK) {
      MS_LOG(ERROR) << "Pre proc failed, fmk " << fmk_type_;
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(WARNING) << "Not support fmk type " << fmk_type_;
  }
  MS_LOG(DEBUG) << "Pre proc graph success.";
  return lite::RET_OK;
}

STATUS AclPassImpl::PostProcGraph(const FuncGraphPtr &func_graph) {
#ifdef ENABLE_ONLINE_MODEL_INFER
  MS_LOG(DEBUG) << "Online model infer no need to change to nhwc format.";
  return lite::RET_OK;
#endif
  if (fmk_type_ == converter::kFmkTypeTf) {
    MS_LOG(DEBUG) << "Tf no need to change to nhwc format.";
    return lite::RET_OK;
  }
  if (!lite::RunOptimizerPass(func_graph, {kToNHWCFormatPass})) {
    MS_LOG(ERROR) << "To NHWC Format failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Post pro graph success.";
  return lite::RET_OK;
}

std::string AclPassImpl::AdjustCnodeName(const PrimitivePtr &prim) {
  std::string name = prim->name();
  if (kAdjustCnodeName.find(name) != kAdjustCnodeName.end()) {
    auto val_ptr = prim->GetAttr(ops::kOriginalOpName);
    if (val_ptr != nullptr) {
      auto origin_name = GetValue<std::string>(val_ptr);
      MS_LOG(DEBUG) << "Change old name " << name << " to new name " << origin_name;
      name = origin_name;
    }
  }
  return name;
}

STATUS AclPassImpl::RunPrimitiveMapper(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph start.";
  MS_ASSERT(func_graph != nullptr);
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto prim = GetCNodePrimitive(cnode);
      CHECK_NULL_RETURN(prim);
      std::string name = AdjustCnodeName(prim);
      auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(name);
      if (mapper == nullptr) {
        MS_LOG(DEBUG) << "Name: " << name << " not need to mapper.";
        continue;
      }
      MS_LOG(INFO) << "Deparser cnode: " << name;
      auto status = mapper->Mapper(cnode);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Deparser primitive failed.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::DeparseGraph(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  if (fmk_type_ == converter::kFmkTypeMs) {
    MS_LOG(INFO) << "MindIr no need to mapper graph";
    return lite::RET_OK;
  }
  if (RunPrimitiveMapper(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Run mapper primitive failed.";
    return lite::RET_ERROR;
  }
  if (lite::AdapteSpatialNode(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adapter spatial node failed.";
    return lite::RET_ERROR;
  }
  if (lite::acl::DelRedundantParameter(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Delete redundant parameter failed.";
    return lite::RET_OK;
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::ConvertGraphToOm(const FuncGraphPtr &func_graph, Buffer *om_data) {
  if (om_data == nullptr) {
    MS_LOG(ERROR) << "Om data is nullptr.";
    return lite::RET_ERROR;
  }
  SetAclModelOptions(func_graph);
  // call interface of cloud
  ModelConverter model_converter;
  model_converter.set_options(options_);
  *om_data = model_converter.LoadMindIR(func_graph);
  if (om_data->Data() == nullptr || om_data->DataSize() == 0) {
    MS_LOG(ERROR) << "Model converter load mindir failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void AclPassImpl::SetAclModelInitOptions(const std::shared_ptr<AscendDeviceInfo> &ascend_info) {
  if (!user_options_cfg_.fusion_switch_config_file_path.empty()) {
    ascend_info->SetFusionSwitchConfigPath(user_options_cfg_.fusion_switch_config_file_path);
  }
  if (!user_options_cfg_.op_select_impl_mode.empty()) {
    ascend_info->SetOpSelectImplMode(user_options_cfg_.op_select_impl_mode);
  }
  if (!user_options_cfg_.buffer_optimize.empty()) {
    ascend_info->SetBufferOptimizeMode(user_options_cfg_.buffer_optimize);
  }
}

void AclPassImpl::SetAclModelBuildOptions(const std::shared_ptr<AscendDeviceInfo> &ascend_info) {
  if (user_options_cfg_.output_type != DataType::kInvalidType) {
    ascend_info->SetOutputType(user_options_cfg_.output_type);
  }
  if (user_options_cfg_.input_shape_map.size() > 0) {
    ascend_info->SetInputShapeMap(user_options_cfg_.input_shape_map);
  }
  if (user_options_cfg_.dynamic_batch_size.size() > 0) {
    ascend_info->SetDynamicBatchSize(user_options_cfg_.dynamic_batch_size);
  }
  if (!user_options_cfg_.dynamic_image_size.empty()) {
    ascend_info->SetDynamicImageSize(user_options_cfg_.dynamic_image_size);
  }
  if (!user_options_cfg_.input_format.empty()) {
    ascend_info->SetInputFormat(user_options_cfg_.input_format);
  }
  if (!user_options_cfg_.input_shape.empty()) {
    ascend_info->SetInputShape(user_options_cfg_.input_shape);
  }
  if (!user_options_cfg_.precision_mode.empty()) {
    ascend_info->SetPrecisionMode(user_options_cfg_.precision_mode);
  }
  if (!user_options_cfg_.insert_op_config_file_path.empty()) {
    ascend_info->SetInsertOpConfigPath(user_options_cfg_.insert_op_config_file_path);
  }
}

std::shared_ptr<mindspore::Context> AclPassImpl::CreateModelContext() {
  auto model_context = std::make_shared<mindspore::Context>();
  if (model_context == nullptr) {
    return nullptr;
  }
  auto ascend_info = std::make_shared<AscendDeviceInfo>();
  if (ascend_info == nullptr) {
    return nullptr;
  }
  ascend_info->SetDeviceID(user_options_cfg_.device_id);
  SetAclModelInitOptions(ascend_info);
  SetAclModelBuildOptions(ascend_info);

  model_context->MutableDeviceInfo().emplace_back(ascend_info);
  return model_context;
}

STATUS AclPassImpl::SetAclModelOptions(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Set acl model options start.";
  auto model_context = CreateModelContext();
  CHECK_NULL_RETURN(model_context);
  options_ = std::make_shared<AclModelOptions>(model_context);
  CHECK_NULL_RETURN(options_);
  auto inputs = func_graph->get_inputs();
  std::vector<std::string> input_names;
  for (auto node : inputs) {
    CHECK_NULL_RETURN(node);
    auto para = node->cast<ParameterPtr>();
    CHECK_NULL_RETURN(para);
    std::string name = para->name();
    for (auto pos = name.find(':'); pos != std::string::npos; pos = name.find(':')) {
      name = name.substr(0, pos) + "_" + name.substr(pos + 1);
      MS_LOG(INFO) << "Input name: " << name;
    }
    para->set_name(name);
    input_names.push_back(name);
  }
  options_->RenameInput(input_names);
  MS_LOG(INFO) << "Set acl model options success.";
  return lite::RET_OK;
}

ParameterPtr AclPassImpl::CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om_data) {
  ParameterPtr om_parameter = func_graph->add_parameter();
  om_parameter->set_name("ACL_om_data");

  auto type_ptr = TypeIdToType(kNumberTypeUInt8);
  ShapeVector shape_vector = {static_cast<int64_t>(om_data.DataSize())};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  om_parameter->set_abstract(abstract_tensor);

  auto param_value =
    std::make_shared<tensor::Tensor>(kNumberTypeUInt8, ShapeVector({static_cast<int64_t>(om_data.DataSize())}));
  auto tensor_data = param_value->data_c();
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "New Tensor failed.";
    return nullptr;
  }
  if (param_value->Size() < om_data.DataSize()) {
    MS_LOG(ERROR) << "Dst buff size  " << param_value->Size() << " should be greater than src buff size "
                  << om_data.DataSize();
    return nullptr;
  }
  if (memcpy_s(tensor_data, param_value->Size(), om_data.Data(), om_data.DataSize()) != EOK) {
    MS_LOG(ERROR) << "Memcpy om data failed.";
    return nullptr;
  }
  om_parameter->set_default_param(param_value);
  return om_parameter;
}

STATUS AclPassImpl::CreateGraphAippInput(const FuncGraphPtr &func_graph, const Buffer &om_data) {
  auto model_process = lite::AclModelProcess(om_data, user_options_cfg_);
  if (model_process.Load() != lite::RET_OK) {
    MS_LOG(ERROR) << "Load om failed.";
    return lite::RET_ERROR;
  }
  std::vector<std::vector<int64_t>> inputs_shape;
  if (model_process.GetInputsShape(&inputs_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get inputs shape failed.";
    return lite::RET_ERROR;
  }
  if (model_process.UnLoad() != lite::RET_OK) {
    MS_LOG(ERROR) << "UnLoad om failed.";
    return lite::RET_ERROR;
  }
  // create aipp parameter
  for (size_t i = 0; i < inputs_shape.size(); i++) {
    ParameterPtr aipp_parameter = func_graph->add_parameter();
    CHECK_NULL_RETURN(aipp_parameter);
    aipp_parameter->set_name("aipp" + std::to_string(i));
    auto type_ptr = TypeIdToType(kNumberTypeUInt8);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, inputs_shape[i]);
    aipp_parameter->set_abstract(abstract_tensor);
    aipp_parameter->set_default_param(nullptr);
    graph_aipp_inputs_.emplace_back(aipp_parameter);
  }
  return lite::RET_OK;
}

// now build the whole graph, not split
STATUS AclPassImpl::BuildGraph(const FuncGraphPtr &func_graph) {
  Buffer om_data;
  if (ConvertGraphToOm(func_graph, &om_data) != lite::RET_OK) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  om_parameter_ = CreateOmParameter(func_graph, om_data);
  if (om_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  if (!user_options_cfg_.insert_op_config_file_path.empty()) {
    if (CreateGraphAippInput(func_graph, om_data) != lite::RET_OK) {
      MS_LOG(ERROR) << "Create aipp input failed.";
      return lite::RET_ERROR;
    }
  }
  MS_LOG(DEBUG) << "Build graph success.";
  return lite::RET_OK;
}

STATUS AclPassImpl::TraceOutput(const AnfNodePtr &node) {
  static size_t iter = 0;
  CHECK_NULL_RETURN(node);
  AnfNodePtr cur_node = node;
  while (cur_node->isa<CNode>() && IsPrimitiveCNode(cur_node, prim::kPrimTupleGetItem)) {
    auto tmp = cur_node->cast<CNodePtr>();
    CHECK_NULL_RETURN(tmp);
    cur_node = tmp->input(kTupleGetItemFirstInputIdx);
  }
  auto cnode = cur_node->cast<CNodePtr>();
  CHECK_NULL_RETURN(cnode);
  std::string name = lite::acl::GetCNodeTargetFuncName(cnode);
  iter++;
  MS_LOG(INFO) << "Func name of cnode " << name << " ,trace iter: " << iter;
  if (name == kMakeTuple) {
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      if (TraceOutput(cnode->input(i)) != lite::RET_OK) {
        MS_LOG(ERROR) << "The input[ " << i << "]"
                      << " trace output failed, name: " << name;
        return lite::RET_ERROR;
      }
    }
  } else if (name == prim::kPrimDepend->name()) {
    if (cnode->inputs().size() < kDependInputNum) {
      MS_LOG(ERROR) << "Length of inputs is " << cnode->inputs().size() << ", which is less than three.";
      return lite::RET_ERROR;
    }
    if (TraceOutput(cnode->input(kDependFirstInputIdx)) != lite::RET_OK) {
      MS_LOG(ERROR) << "Depend node trace output failed.";
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(INFO) << "Graph out name: " << cnode->fullname_with_scope();
    graph_output_names_.emplace_back(cnode->fullname_with_scope());
    std::vector<int64_t> dims;
    if (lite::acl::GetShapeVectorFromCNode(cnode, &dims) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get node shape failed.";
      return lite::RET_ERROR;
    }
    graph_output_dims_.emplace_back(dims);
    graph_outputs_.emplace_back(cnode);
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::GetFuncGraphOutputInfo(const FuncGraphPtr &func_graph) {
  AnfNodePtr return_input = func_graph->output();
  CHECK_NULL_RETURN(return_input);
  if (TraceOutput(return_input) != lite::RET_OK) {
    MS_LOG(ERROR) << "Trace output failed.";
    return lite::RET_ERROR;
  }
  if (graph_outputs_.empty() || graph_outputs_.size() != graph_output_dims_.size()) {
    MS_LOG(ERROR) << "Graph output size is error, num size: " << graph_outputs_.size()
                  << " dim size: " << graph_output_dims_.size();
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS AclPassImpl::SetMultiOutputs(const CNodePtr &new_cnode, TypeId data_type) {
  AbstractBasePtrList abstract_list;
  for (size_t j = 0; j < graph_outputs_.size(); j++) {
    auto abstract_tensor = lite::CreateTensorAbstract(graph_output_dims_[j], data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract tensor is nullptr for output " << j;
      return lite::RET_ERROR;
    }
    abstract_list.emplace_back(abstract_tensor);
  }
  new_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return lite::RET_OK;
}

STATUS AclPassImpl::SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  STATUS ret = GetFuncGraphOutputInfo(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get output info of graph failed.";
    return lite::RET_ERROR;
  }
  custom_node->AddAttr(kOutputNames, MakeValue(graph_output_names_));
  TypeId type = lite::acl::GetTypeFromNode(graph_outputs_[0]);
  if (graph_outputs_.size() == 1) {
    auto abstract_tensor = lite::CreateTensorAbstract(graph_output_dims_[0], type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract_tensor is nullptr.";
      return lite::RET_ERROR;
    }
    custom_node->set_abstract(abstract_tensor);
    return lite::RET_OK;
  }
  if (SetMultiOutputs(custom_node, type) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set multi graph output failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void AclPassImpl::SetCustomAttrs(const std::shared_ptr<ops::Custom> &prim) {
  // add output_shape attr
  std::string output_dim_str;
  for (const auto &item : graph_output_dims_) {
    output_dim_str += std::to_string(item.size()) + ",";
    for (const auto &val : item) {
      output_dim_str += std::to_string(val) + ",";
    }
  }
  std::vector<uint8_t> output_dim_char(output_dim_str.begin(), output_dim_str.end());
  std::map<std::string, std::vector<uint8_t>> attrs = {{lite::acl::kOutputShapes, output_dim_char}};
  prim->set_attr(attrs);
}

CNodePtr AclPassImpl::CreateCustomNode(const FuncGraphPtr &func_graph) {
  auto prim = std::make_shared<mindspore::ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "New custom op failed.";
    return nullptr;
  }
  prim->set_type(kCustomPrimTypeACL);
  auto graph_input = func_graph->get_inputs();
  CNodePtr custom_node = func_graph->NewCNode(prim, graph_input);
  if (custom_node == nullptr) {
    MS_LOG(ERROR) << "Custom cnode failed.";
    return nullptr;
  }
  custom_node->set_fullname_with_scope(kCustomNodeName);
  custom_node->add_input(om_parameter_);

  if (SetCustomOutputs(func_graph, custom_node) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set custom outputs failed.";
    return nullptr;
  }
  SetCustomAttrs(prim);
  return custom_node;
}

STATUS AclPassImpl::ReplaceInputsByAippInputs(const FuncGraphPtr &func_graph) {
  auto graph_inputs = func_graph->get_inputs();
  if (graph_aipp_inputs_.size() != graph_inputs.size()) {
    MS_LOG(ERROR) << "Input size of aipp " << graph_aipp_inputs_.size() << " and input size of func graph "
                  << graph_inputs.size() << " are not equal.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto aipp_abstract = graph_aipp_inputs_[i]->abstract();
    CHECK_NULL_RETURN(aipp_abstract);
    graph_inputs[i]->set_abstract(aipp_abstract->Clone());
  }
  return lite::RET_OK;
}

STATUS AclPassImpl::ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                            const CNodePtr &custom_node) {
  if (graph_outputs_.size() == 1) {
    if (!manager->Replace(graph_outputs_[0], custom_node)) {
      MS_LOG(ERROR) << "Replace node failed.";
      return lite::RET_ERROR;
    }
  } else {
    for (size_t j = 0; j < graph_outputs_.size(); ++j) {
      auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "New TupleGetItem failed for output " << j;
        return lite::RET_ERROR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(j));
      AnfNodePtrList inputs{tuple_get_item_prim, custom_node, get_item_value};
      CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "New get item cnode failed for output " << j;
        return lite::RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(custom_node->fullname_with_scope() + "_getitem_" + std::to_string(j));
      if (!manager->Replace(graph_outputs_[j], get_item_cnode)) {
        MS_LOG(ERROR) << "Replace node failed for output " << j;
        return lite::RET_ERROR;
      }
    }
  }
  if (!graph_aipp_inputs_.empty()) {
    if (ReplaceInputsByAippInputs(func_graph) != lite::RET_OK) {
      MS_LOG(ERROR) << "Replace inputs by aipp inputs failed.";
      return lite::RET_ERROR;
    }
  }
  MS_LOG(DEBUG) << "Modify graph by custom node success.";
  return lite::RET_OK;
}

bool AclPassImpl::Run(const FuncGraphPtr &func_graph) {
  if (!IsDeviceAscend()) {
    MS_LOG(INFO) << "The device is not ascend, no need to pass.";
    return true;
  }
  MS_LOG(INFO) << "Acl pass run start.";
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func_graph is nullptr.";
    return false;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr.";
    return false;
  }

  if (PreProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Pre proc graph failed.";
    return false;
  }

  if (DeparseGraph(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Deparse graph failed.";
    return false;
  }

  if (BuildGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Build graph failed.";
    return false;
  }

  custom_node_ = CreateCustomNode(func_graph);
  if (custom_node_ == nullptr) {
    MS_LOG(ERROR) << "Create custom node failed.";
    return false;
  }
  // prepare graph for export create
  if (ModifyGraphByCustomNode(func_graph, manager, custom_node_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Modify func graph by custom failed.";
    return false;
  }

  if (PostProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Post proc graph failed.";
    return false;
  }
  MS_LOG(INFO) << "Acl pass run end.";
  return true;
}
}  //  namespace opt
}  // namespace mindspore
