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

#include "tools/converter/parser/pytorch/pytorch_model_parser.h"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/inliner.h"
#include "torch/csrc/jit/passes/normalize_ops.h"
#include "include/registry/node_parser_registry.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "src/common/file_utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"

using mindspore::converter::kFmkTypePytorch;
namespace mindspore {
namespace lite {
api::FuncGraphPtr PytorchModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file;
  NotSupportOp::GetInstance()->set_fmk_type("PYTORCH");
  auto anf_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(anf_graph != nullptr, nullptr, "create FuncGraph failed");
  res_graph_ = api::MakeShared<api::FuncGraph>(anf_graph);
  MS_CHECK_TRUE_MSG(res_graph_ != nullptr, nullptr, "create FuncGraph failed");
  auto status = InitOriginModel(model_file);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "init origin model failed.";
    return nullptr;
  }

  status = ConvertTorchGraph(anf_graph);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert pytorch graph failed.";
    return nullptr;
  }
  static auto root_func_manager = Manage(anf_graph);
  MS_ASSERT(root_func_manager != nullptr);
  for (auto &subgraph : all_subgraphs_) {
    MS_ASSERT(subgraph != nullptr);
    subgraph->set_manager(root_func_manager);
    subgraph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypePytorch)));
  }
  anf_graph->set_attr("graph_name", MakeValue("main_graph"));
  anf_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypePytorch)));
  if ((status = CommonAnfAdjust(anf_graph)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  auto unify_format = std::make_shared<UnifyFormatToNHWC>(kFmkTypePytorch, false, flag.save_type);
  MS_CHECK_TRUE_MSG(unify_format != nullptr, nullptr, "create unify_format return nullptr");
  if (!unify_format->Run(anf_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  return res_graph_;
}
STATUS PytorchModelParser::InitOriginModel(const std::string &model_file) {
  if (ValidateFileStr(model_file, ".pt") != RET_OK && ValidateFileStr(model_file, ".pth") != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.pt or *.pth";
    return RET_ERROR;
  }
  std::string model_path = RealPath(model_file.c_str());
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Binary proto file path " << model_file << " is not valid";
    return RET_ERROR;
  }
  // only linux supports to convert pytorch model.
  if (access(model_path.c_str(), F_OK) != 0 || access(model_path.c_str(), R_OK) != 0) {
    MS_LOG(ERROR) << "The pytorch model file is not exist or can't be read.";
    return RET_ERROR;
  }

  auto module = torch::jit::load(model_path);
  module.eval();                               // eval to expand function call
  module = torch::jit::freeze_module(module);  // freeze module
  torch_model_ = module.get_method("forward").graph();
  CHECK_NULL_RETURN(torch_model_);
  // parse submodules in graph
  torch::jit::Inline(*torch_model_);
  torch::jit::NormalizeOps(torch_model_);
  return RET_OK;
}

STATUS PytorchModelParser::ConvertTorchGraph(const FuncGraphPtr &anf_graph) {
  MS_ASSERT(torch_graph != nullptr && anf_graph != nullptr && anf_nodes_map != nullptr &&
            extra_subgraph_inputs != nullptr);
  STATUS status = ConvertGraphInputs(anf_graph);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph inputs failed.";
    return RET_OK;
  }

  status = ConvertNodes(anf_graph);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert nodes failed.";
    return RET_ERROR;
  }

  status = ConvertGraphOutputs(anf_graph);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph outputs failed.";
    return RET_ERROR;
  }
  return status;
}

STATUS PytorchModelParser::ConvertGraphInputs(const FuncGraphPtr &anf_graph) {
  MS_ASSERT(anf_graph != nullptr && anf_nodes_map != nullptr);
  for (auto &input : torch_model_->inputs()) {
    auto input_name = input->debugName();
    if (anf_nodes_map_.find(input_name) != anf_nodes_map_.end()) {
      continue;
    }
    auto type = input->type();
    MS_CHECK_TRUE_RET(type != nullptr, RET_ERROR);
    auto tensor_type = type->cast<at::TensorType>();
    if (tensor_type == nullptr) {
      MS_LOG(DEBUG) << "The input is not a tensor, but a: " << c10::typeKindToString(type->kind());
      continue;
    }
    auto scalar_type = tensor_type->scalarType().value_or(at::ScalarType::Float);
    auto data_type = PytorchNodeParser::GetDataTypeFromTorch(scalar_type);
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "not support pytorch data type " << scalar_type;
      return RET_ERROR;
    }
    std::vector<int64_t> input_shape = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(input_name);
    if (input_shape.empty()) {
      if (tensor_type->sizes().isComplete()) {
        input_shape = tensor_type->sizes().concrete_sizes().value();
      } else {
        MS_LOG(WARNING) << "The input shape is empty.";
      }
    }
    auto abstract_tensor = CreateTensorAbstract(input_shape, data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    auto parameter = anf_graph->add_parameter();
    MS_CHECK_TRUE_MSG(parameter != nullptr, RET_NULL_PTR, "create parameter return nullptr");
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(input_name);
    anf_nodes_map_.emplace(input_name, parameter);
  }
  return RET_OK;
}

STATUS BuildReturnNode(const FuncGraphPtr &anf_graph, const std::vector<AnfNodePtr> &return_inputs) {
  MS_ASSERT(anf_graph != nullptr);
  auto return_prim_ptr = std::make_shared<ops::Return>();
  if (return_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "new Return failed";
    return RET_NULL_PTR;
  }
  auto return_prim = return_prim_ptr->GetPrim();
  MS_CHECK_TRUE_RET(return_prim != nullptr, RET_ERROR);
  auto return_cnode = anf_graph->NewCNode(return_prim, return_inputs);
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  return_cnode->set_fullname_with_scope("Return");
  anf_graph->set_return(return_cnode);
  return RET_OK;
}

STATUS PytorchModelParser::ConvertGraphOutputs(const FuncGraphPtr &anf_graph) {
  MS_ASSERT(anf_graph != nullptr);
  std::vector<AnfNodePtr> return_inputs;
  if (torch_model_->outputs().size() == 0) {
    MS_LOG(ERROR) << "pytorch graph has no output";
    return RET_ERROR;
  }
  if (torch_model_->outputs().size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    for (const auto &output : torch_model_->outputs()) {
      auto output_name = output->debugName();
      if (anf_nodes_map_.find(output_name) == anf_nodes_map_.end()) {
        MS_LOG(ERROR) << "graph output get failed.";
        return RET_ERROR;
      }
      auto cnode = anf_nodes_map_.at(output_name);
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_prim = make_tuple_prim_ptr->GetPrim();
    MS_CHECK_TRUE_RET(make_tuple_prim != nullptr, RET_ERROR);
    auto make_tuple_cnode = anf_graph->NewCNode(make_tuple_prim, make_tuple_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    return_inputs.emplace_back(make_tuple_cnode);
  } else {
    const auto &output = torch_model_->outputs().front();
    if (anf_nodes_map_.find(output->debugName()) == anf_nodes_map_.end()) {
      MS_LOG(ERROR) << "graph output get failed.";
      return RET_ERROR;
    }
    auto cnode = anf_nodes_map_.at(output->debugName());
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    return_inputs.emplace_back(cnode);
  }
  if (BuildReturnNode(anf_graph, return_inputs) != RET_OK) {
    MS_LOG(ERROR) << "build return node failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS CopyDataFromTorchTensor(char *dst_data, const at::Tensor &torch_tensor, TypeId data_type) {
  auto ele_size = abstract::TypeIdSize(data_type);
  MS_CHECK_TRUE_RET(ele_size > 0, RET_ERROR);
  auto data_shape = torch_tensor.sizes().vec();
  auto stride = torch_tensor.strides().vec();
  if (data_shape.empty()) {
    auto data_size = torch_tensor.numel() * ele_size;
    data_shape.push_back(data_size);
    stride.push_back(1);
  }
  char *data_ptr = reinterpret_cast<char *>(torch_tensor.data_ptr());
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "The tensor data is nullptr.";
    return RET_ERROR;
  }
  size_t idx = 0;
  std::function<void(size_t, size_t)> copy_data = [&](size_t dim, size_t offset) {
    if (dim == data_shape.size() - 1) {
      for (int i = 0; i < data_shape[dim]; i++) {
        auto src_ptr = data_ptr + offset + i * stride[dim] * ele_size;
        auto dst_ptr = dst_data + (idx++) * ele_size;
        MS_CHECK_TRUE_RET_VOID(memcpy_s(dst_ptr, ele_size, src_ptr, ele_size) == EOK);
      }
    } else {
      for (int i = 0; i < data_shape[dim]; i++) {
        copy_data(dim + 1, offset + i * stride[dim] * ele_size);
      }
    }
  };
  copy_data(0, 0);
  return RET_OK;
}

STATUS ConvertConstNode(const torch::jit::Node *torch_node, const FuncGraphPtr &anf_graph,
                        std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map) {
  ParameterPtr parameter = nullptr;
  auto output = torch_node->output();
  auto type_kind = output->type()->kind();
  auto value = torch::jit::toIValue(output);
  MS_CHECK_TRUE_RET(value.has_value(), RET_ERROR);
  switch (type_kind) {
    case c10::TypeKind::BoolType: {
      auto data = static_cast<int>(value.value().toBool());
      parameter = opt::BuildIntValueParameterNode(anf_graph, data, output->debugName());
    } break;
    case c10::TypeKind::IntType: {
      auto data = static_cast<int>(value.value().toInt());
      parameter = opt::BuildIntValueParameterNode(anf_graph, data, output->debugName());
    } break;
    case c10::TypeKind::FloatType: {
      auto data = static_cast<float>(value.value().toDouble());
      parameter = opt::BuildFloatValueParameterNode(anf_graph, data, output->debugName());
    } break;
    case c10::TypeKind::ListType: {
      auto element_type = value->toList().elementType()->kind();
      switch (element_type) {
        case c10::TypeKind::IntType: {
          auto ori_data = value.value().toIntVector();
          std::vector<int> data;
          std::transform(ori_data.begin(), ori_data.end(), std::back_inserter(data),
                         [](int64_t ele) { return static_cast<int>(ele); });
          parameter = opt::BuildIntVecParameterNode(anf_graph, data, output->debugName());
        } break;
        case c10::TypeKind::FloatType: {
          auto ori_data = value.value().toDoubleVector();
          std::vector<float> data;
          std::transform(ori_data.begin(), ori_data.end(), std::back_inserter(data),
                         [](double ele) { return static_cast<float>(ele); });
          parameter = opt::BuildFloatVecParameterNode(anf_graph, data, output->debugName());
        } break;
        default:
          MS_LOG(ERROR) << "Unsupported data type: " << c10::typeKindToString(element_type);
          return RET_ERROR;
      }
    } break;
    case c10::TypeKind::TensorType: {
      auto torch_tensor = value.value().toTensor();
      auto data_type = PytorchNodeParser::GetDataTypeFromTorch(torch_tensor.scalar_type());
      auto data_size = torch_tensor.numel() * abstract::TypeIdSize(data_type);
      char *data_ptr = reinterpret_cast<char *>(malloc(data_size));
      if (data_ptr == nullptr) {
        MS_LOG(ERROR) << "malloc data failed.";
        return RET_ERROR;
      }
      if (CopyDataFromTorchTensor(data_ptr, torch_tensor, data_type) != RET_OK) {
        MS_LOG(ERROR) << "Copy data from torch tensor failed.";
        free(data_ptr);
        return RET_ERROR;
      }
      auto data_shape = torch_tensor.sizes().vec();
      auto tensor_info = CreateTensorInfo(data_ptr, data_size, data_shape, data_type);
      free(data_ptr);
      if (tensor_info == nullptr) {
        MS_LOG(ERROR) << "Create tensorInfo failed.";
        return RET_ERROR;
      }
      parameter = opt::BuildParameterNode(anf_graph, tensor_info, output->debugName());
    } break;
    case c10::TypeKind::NoneType:
      MS_LOG(DEBUG) << "The const node is none.";
      return RET_OK;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << c10::typeKindToString(type_kind);
      return RET_ERROR;
  }
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "The parameter is nullptr.";
    return RET_ERROR;
  }
  anf_nodes_map->emplace(output->debugName(), parameter);
  return RET_OK;
}

STATUS BuildOpInputs(const torch::jit::Node *torch_node, std::vector<AnfNodePtr> *op_inputs,
                     const std::vector<size_t> &input_indices,
                     const std::unordered_map<std::string, AnfNodePtr> &anf_nodes_map) {
  MS_ASSERT(torch_node != nullptr && op_inputs != nullptr);
  for (size_t idx : input_indices) {
    auto input = torch_node->input(idx);
    MS_CHECK_TRUE_RET(input != nullptr, RET_ERROR);
    auto input_name = input->debugName();
    if (input_name.empty()) {
      continue;
    }

    if (anf_nodes_map.find(input_name) != anf_nodes_map.end()) {
      op_inputs->push_back(anf_nodes_map.at(input_name));
    } else {
      MS_LOG(ERROR) << "could not find input node: " << input_name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS BuildOpOutputs(const torch::jit::Node *torch_node, const FuncGraphPtr &anf_graph,
                      std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, const CNodePtr &cnode) {
  MS_ASSERT(torch_node != nullptr && anf_graph != nullptr && cnode != nullptr && anf_nodes_map != nullptr);
  if (torch_node->outputs().size() == 1) {
    auto abstract_tensor = CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    cnode->set_abstract(abstract_tensor);
    anf_nodes_map->emplace(torch_node->output()->debugName(), cnode);
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (const auto &output : torch_node->outputs()) {
      auto abstract_tensor = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (abstract_tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      abstract_list.emplace_back(abstract_tensor);
      auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim = tuple_get_item_prim_ptr->GetPrim();
      MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, RET_NULL_PTR, "get prim return nullptr");
      auto tuple_get_item = NewValueNode(tuple_get_item_prim);
      MS_CHECK_TRUE_MSG(tuple_get_item != nullptr, RET_NULL_PTR, "create ValueNode return nullptr");
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      MS_CHECK_TRUE_MSG(get_item_value != nullptr, RET_NULL_PTR, "create ValueNode return nullptr");
      std::vector<AnfNodePtr> inputs{tuple_get_item, cnode, get_item_value};
      CNodePtr get_item_cnode = anf_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_getitem_" + std::to_string(op_idx));
      auto get_item_abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (get_item_abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      get_item_cnode->set_abstract(get_item_abstract);
      anf_nodes_map->emplace(output->debugName(), get_item_cnode);
      op_idx++;
    }
    auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
    CHECK_NULL_RETURN(new_abstract_list);
    cnode->set_abstract(new_abstract_list);
  }
  anf_nodes_map->emplace(torch_node->kind().toUnqualString(), cnode);
  return RET_OK;
}

STATUS PytorchModelParser::ConvertNodes(const FuncGraphPtr &anf_graph) {
  MS_ASSERT(anf_graph != nullptr);
  STATUS status = RET_OK;
  for (const auto &torch_node : torch_model_->nodes()) {
    ops::PrimitiveCPtr primitive_c = nullptr;
    auto node_type = PytorchNodeParser::GetTorchNodeType(torch_node);
    MS_CHECK_TRUE_RET(!node_type.empty(), RET_ERROR);
    // convert constant node.
    if (node_type == "Constant") {
      if (ConvertConstNode(torch_node, anf_graph, &anf_nodes_map_) != RET_OK) {
        MS_LOG(ERROR) << "Convert constant node failed.";
        return RET_ERROR;
      }
      continue;
    }

    std::vector<AnfNodePtr> op_inputs;
    std::vector<size_t> input_indices;
    auto node_parser_builtin = PytorchNodeParserRegistry::GetInstance().GetNodeParser(node_type);
    if (node_parser_builtin == nullptr) {
      NotSupportOp::GetInstance()->InsertOp(node_type);
      status = status == RET_OK ? RET_NOT_FIND_OP : status;
      MS_LOG(ERROR) << "not support pytorch op type " << node_type;
      continue;
    }
    MS_LOG(INFO) << "parse op:" << node_type;
    primitive_c = node_parser_builtin->Parse(torch_node, &input_indices);
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "parse node " << node_type << " failed.";
      status = RET_ERROR;
      continue;
    }
    // set default format and input indices.
    if (primitive_c->GetAttr(ops::kOriginalFormat) == nullptr) {
      primitive_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(NCHW));
    }
    if (input_indices.empty()) {
      input_indices.resize(torch_node->inputs().size());
      std::iota(input_indices.begin(), input_indices.end(), 0);
    }

    if (BuildOpInputs(torch_node, &op_inputs, input_indices, anf_nodes_map_) != RET_OK) {
      MS_LOG(ERROR) << "BuildOpInputs failed.";
      return RET_ERROR;
    }
    auto new_cnode = anf_graph->NewCNode(primitive_c, op_inputs);
    if (new_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    new_cnode->set_fullname_with_scope(std::string(torch_node->kind().toUnqualString()) + "_" +
                                       torch_node->output(0)->debugName());
    if (BuildOpOutputs(torch_node, anf_graph, &anf_nodes_map_, new_cnode) != RET_OK) {
      MS_LOG(ERROR) << "BuildOpOutputs failed.";
      return RET_ERROR;
    }
  }
  return status;
}

REG_MODEL_PARSER(kFmkTypePytorch, LiteModelParserCreator<PytorchModelParser>)
}  // namespace lite
}  // namespace mindspore
