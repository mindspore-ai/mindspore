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

#include "backend/optimizer/trt_pass/trt_converter_context.h"

#include "runtime/device/gpu/trt_loader.h"
#include "backend/optimizer/trt_pass/trt_op_factory.h"
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"
#include "utils/convert_utils.h"
#include "utils/utils.h"
#include "utils/singleton.h"
#include "utils/ms_context.h"

namespace mindspore::opt {
bool TrtConverterContext::Init() {
  auto trt_loader = Singleton<device::gpu::TrtLoader>::Instance();
  builder_ = trt_loader.CreateInferBuilder(&Singleton<TrtLogger>::Instance());
  MS_EXCEPTION_IF_NULL(builder_);

  auto batch_type = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network_ = TrtPtr(builder_->createNetworkV2(batch_type));
  MS_EXCEPTION_IF_NULL(network_);

  config_ = TrtPtr(builder_->createBuilderConfig());
  MS_EXCEPTION_IF_NULL(config_);

  InitInputTable();
  InitValueNodeTable();
  return true;
}

bool TrtConverterContext::Parser() {
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph_->get_return());
  const auto &converter_factory = TrtOpFactory::GetInstance();
  for (auto node : node_list) {
    if (!node->isa<CNode>()) {
      continue;
    }

    // Transform AnfNode To Trt layer.
    // Bypass control node including Depend, Load, UpdateState, TupleGetItem, MakeTuple.
    std::string op_name = AnfAlgo::GetCNodePrimitive(node)->name();
    if (!AnfAlgo::IsRealKernel(node) && op_name != "Return") {
      continue;
    }

    ConvertFunc convert_func = converter_factory.GetConvertFunc(op_name);
    auto result = convert_func(node, this->shared_from_this());
    if (!result.first) {
      MS_LOG(WARNING) << op_name << " converter failed.";
      return false;
    }
    auto ret = StoreLayerOutput(node, result.second);
    if (!ret) {
      MS_LOG(WARNING) << op_name << " converter failed.";
      return false;
    }
  }

  return true;
}

bool TrtConverterContext::Serialize(std::string *model) {
  MS_EXCEPTION_IF_NULL(model);
  builder_->setMaxBatchSize(batch_size_);
  config_->setMaxWorkspaceSize(workspace_size_);

  // Set precision mode
  const auto &context = MsContext::GetInstance();
  const auto &precision_mode = context->get_param<std::string>(MS_CTX_INFER_PRECISION_MODE);
  if (precision_mode == "fp16") {
    MS_LOG(INFO) << "Inference with mixed precision mode";
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  MS_LOG(WARNING) << "It will take few minutes for operators selection.";
  engine_ = TrtPtr(builder_->buildEngineWithConfig(*network_, *config_));
  MS_EXCEPTION_IF_NULL(engine_);

  std::shared_ptr<nvinfer1::IHostMemory> model_data = TrtPtr(engine_->serialize());
  *model = string(static_cast<const char *>(model_data->data()), model_data->size());
  return true;
}

bool TrtConverterContext::InitInputTable() {
  const std::vector<AnfNodePtr> graph_inputs = func_graph_->parameters();
  for (auto input_node : graph_inputs) {
    if (!input_node->isa<Parameter>()) {
      continue;
    }

    auto input = input_node->cast<ParameterPtr>();
    if (AnfAlgo::IsParameterWeight(input)) {
      const auto &param_value = input->default_param();
      MS_EXCEPTION_IF_NULL(param_value);
      auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_value);
      MS_EXCEPTION_IF_NULL(tensor);

      nvinfer1::Weights weight;
      weight.values = tensor->data_c();
      std::variant<bool, nvinfer1::DataType> type = TrtUtils::MsDtypeToTrtDtype(tensor->data_type());
      TRT_VARIANT_CHECK(type, 1UL, false);
      weight.type = std::get<nvinfer1::DataType>(type);
      weight.count = tensor->DataSize();
      output_map_[input_node][0] = LayerInput(weight, tensor->shape());
    }
  }
  return true;
}

bool TrtConverterContext::InitValueNodeTable() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  const std::vector<AnfNodePtr> &node_list = TopoSort(func_graph_->get_return());
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
      auto value_node = node->cast<ValueNodePtr>();
      auto &node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);

      if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
        std::vector<tensor::TensorPtr> tensors;
        TensorValueToTensor(node_value, &tensors);
        for (size_t i = 0; i < tensors.size(); i++) {
          const auto &tensor = tensors[i];
          nvinfer1::Weights weight;
          weight.values = tensor->data_c();
          std::variant<bool, nvinfer1::DataType> type = TrtUtils::MsDtypeToTrtDtype(tensor->data_type());
          TRT_VARIANT_CHECK(type, 1UL, false);
          weight.type = std::get<nvinfer1::DataType>(type);
          weight.count = tensor->DataSize();
          output_map_[value_node][i] = LayerInput(weight, tensor->shape());
        }
      }
    }
  }
  return true;
}

bool TrtConverterContext::StoreLayerOutput(const AnfNodePtr &node, const std::vector<nvinfer1::ITensor *> &nv_tensors) {
  if (nv_tensors.size() != AnfAlgo::GetOutputTensorNum(node)) {
    MS_LOG(INFO) << node->DebugString() << " output num not match. expect: " << AnfAlgo::GetOutputTensorNum(node)
                 << ", while got: " << nv_tensors.size();
  }

  for (size_t tensor_index = 0; tensor_index < nv_tensors.size(); ++tensor_index) {
    if (nv_tensors[tensor_index] != nullptr) {
      const nvinfer1::Dims &dim = nv_tensors[tensor_index]->getDimensions();
      const std::vector<int64_t> &shape = TrtUtils::TrtDimsToMsDims(dim);
      output_map_[node][tensor_index] = LayerInput(nv_tensors[tensor_index], shape);

      std::ostringstream oss;
      oss << node->fullname_with_scope() << ", output: " << tensor_index << ": [ ";
      for (int32_t dim_index = 0; dim_index < dim.nbDims; dim_index++) {
        oss << dim.d[dim_index] << " ";
      }
      oss << "]";
      MS_LOG(INFO) << oss.str();
    }
  }
  return true;
}

LayerInput *TrtConverterContext::LoadInputOnDemand(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto input = node->cast<ParameterPtr>();
  std::variant<bool, nvinfer1::DataType> type = TrtUtils::MsDtypeToTrtDtype(AnfAlgo::GetOutputInferDataType(node, 0));
  TRT_VARIANT_CHECK(type, 1UL, nullptr);
  const auto &trt_dtype = std::get<nvinfer1::DataType>(type);
  const nvinfer1::Dims &trt_dims = TrtUtils::MsDimsToTrtDims(AnfAlgo::GetOutputInferShape(node, 0), false);
  nvinfer1::ITensor *tensor = network_->addInput(input->name().c_str(), trt_dtype, trt_dims);
  const std::vector<int64_t> &shape = TrtUtils::TrtDimsToMsDims(trt_dims);
  output_map_[node][0] = LayerInput(tensor, shape);
  return &output_map_[node][0];
}

bool TrtConverterContext::LoadLayerInput(const AnfNodePtr &node, std::vector<LayerInput> *inputs) {
  std::vector<session::KernelWithIndex> real_inputs;
  AnfAlgo::GetRealInputs(node, &real_inputs);
  for (auto item : real_inputs) {
    auto node_iter = output_map_.find(item.first);
    if (node_iter == output_map_.end()) {
      if (item.first->isa<Parameter>()) {
        LayerInput *input = LoadInputOnDemand(item.first);
        if (input == nullptr) {
          MS_LOG(WARNING) << "LoadLayerInput failed.";
          return false;
        }
        inputs->push_back(*input);
        continue;
      }
      MS_LOG(WARNING) << "node: " << node->DebugString() << " not found.";
      return false;
    }

    auto out_iter = node_iter->second.find(item.second);
    if (out_iter == node_iter->second.end()) {
      MS_LOG(WARNING) << "node: " << node->DebugString() << "output index: " << item.second << " not found.";
      return false;
    }

    inputs->push_back(out_iter->second);
  }
  return true;
}

std::vector<AnfNodePtr> TrtConverterContext::GetGraphInputs() const {
  // Get Anf-graph inputs without weights. All weights were binded to Trt-graph.
  std::unordered_map<std::string, AnfNodePtr> graph_inputs;
  for (const auto &input_node : func_graph_->parameters()) {
    if (!input_node->isa<Parameter>()) {
      continue;
    }

    auto input = input_node->cast<ParameterPtr>();
    if (!AnfAlgo::IsParameterWeight(input)) {
      graph_inputs.insert(std::make_pair(input->name(), input_node));
    }
  }

  // Keep the graph inputs in order of the binding name.
  std::vector<AnfNodePtr> trt_inputs;
  for (int32_t i = 0; i < engine_->getNbBindings(); ++i) {
    if (!engine_->bindingIsInput(i)) {
      continue;
    }
    auto iter = graph_inputs.find(engine_->getBindingName(i));
    if (iter == graph_inputs.end()) {
      MS_LOG(EXCEPTION) << "Get graph inputs failed. input name" << engine_->getBindingName(i);
    }
    trt_inputs.push_back(iter->second);
  }
  return trt_inputs;
}

std::tuple<std::map<size_t, size_t>, std::vector<session::KernelWithIndex>> TrtConverterContext::GetGraphOutputs()
  const {
  std::vector<session::KernelWithIndex> anf_output_list;
  AnfAlgo::GetRealInputs(func_graph_->get_return(), &anf_output_list);

  std::map<size_t, size_t> anf_trt_index_map;
  std::vector<session::KernelWithIndex> trt_output_list(anf_output_list.size());
  size_t trt_index = 0;
  for (int32_t i = 0; i < engine_->getNbBindings(); ++i) {
    if (!engine_->bindingIsInput(i)) {
      const std::string &name = engine_->getBindingName(i);
      size_t pos = name.find_first_not_of("return_output_");
      size_t anf_index = atoi(name.substr(pos).c_str());

      anf_trt_index_map.insert(std::make_pair(anf_index, trt_index));
      trt_output_list[trt_index] = anf_output_list[anf_index];
      trt_index++;
    }
  }

  return std::make_tuple(anf_trt_index_map, trt_output_list);
}

std::shared_ptr<tensor::Tensor> TrtConverterContext::CreateTempWeight(const TypeId &type,
                                                                      const std::vector<size_t> &shape) {
  ShapeVector shape_int;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_int), SizeToLong);
  auto tensor = std::make_shared<tensor::Tensor>(type, shape_int);
  temp_weights_.push_back(tensor);
  return tensor;
}
}  // namespace mindspore::opt
