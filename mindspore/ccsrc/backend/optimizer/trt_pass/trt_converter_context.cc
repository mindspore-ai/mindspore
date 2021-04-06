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

#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <algorithm>
#include "runtime/device/gpu/trt_loader.h"
#include "backend/optimizer/trt_pass/trt_op_factory.h"
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"
#include "utils/convert_utils.h"
#include "utils/utils.h"
#include "utils/singleton.h"

namespace mindspore::opt {
namespace {
void GetRealOutputRecursively(const AnfNodePtr &node, size_t output_index,
                              std::vector<session::KernelWithIndex> *inputs) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>() || node->isa<Parameter>()) {
    return inputs->push_back(std::make_pair(node, 0));
  }

  // Skip control node
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) || AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad) ||
      AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    return GetRealOutputRecursive(node->cast<CNodePtr>()->input(kRealInputIndexInDepend), 0, inputs);
  }

  // Bypass TupleGetItem
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    auto tuple_get_item = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_item);
    auto input = AnfAlgo::GetTupleGetItemRealInput(tuple_get_item);
    auto index = AnfAlgo::GetTupleGetItemOutIndex(tuple_get_item);

    // Conceal MakeTuple + TupleGetItem pair.
    if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimMakeTuple)) {
      auto make_tuple = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple);
      auto real_input = AnfAlgo::GetInputNode(make_tuple, index);
      return GetRealOutputRecursive(real_input, 0, inputs);
    }

    // Skip TupleGetItem.
    return GetRealOutputRecursive(input, index, inputs);
  }

  // Flatten MakeTuple inputs.
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    auto make_tuple = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t input_num = AnfAlgo::GetInputTensorNum(make_tuple);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      auto input_node = AnfAlgo::GetInputNode(make_tuple, input_index);
      GetRealOutputRecursive(input_node, 0, inputs);
    }
    return;
  }

  return inputs->push_back(std::make_pair(node, output_index));
}

/* Get node real inputs bypass control nodes.
 *   Examples:
 *     Case 1:
 *       c = Conv2D(a, b)
 *       d = ReLU(c)
 *     result: d--> (c)
 *
 *     Case 2:
 *       c = Conv2D(a, b)
 *       d = Depend(c, v)
 *       e = ReLU(d)
 *     result: d -> (c)
 *
 *     Case 3:
 *       (f, g, h, i, j) = BatchNorm(a, b, c, d, e)
 *       k = TupleGetItem((f, g, h, i, j), 0)
 *       l = ReLU(k)
 *     result: l -> (f)
 *
 *     Case 4:
 *       c = Conv2D(a, b)
 *       e = MakeTuple(c, d)
 *       f = TupleGetItem(e, 0)
 *       g = ReLU(k)
 *     result: g -> (c)
 *
 *     Case 5:
 *       b = MakeTuple(a1, a2, a3)
 *       c = MakeTuple(b, a4)
 *       d = return(c)
 *     result d -> (a1, a2, a3, a4)
 */
void GetRealInputs(const AnfNodePtr &node, std::vector<session::KernelWithIndex> *inputs) {
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), input_index);
    GetRealOutputRecursively(input_node, 0, inputs);
  }
}
}  // namespace

bool TrtConverterContext::Init() {
  auto trt_loader = Singleton<device::gpu::TrtLoader>::Instance();
  builder_ = trt_loader.CreateInferBuilder(&Singleton<TrtLogger>::Instance());
  MS_EXCEPTION_IF_NULL(builder_);

  auto batch_type = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network_ = TrtPtr(builder_->createNetworkV2(batch_type));
  MS_EXCEPTION_IF_NULL(network_);

  config_ = TrtPtr(builder_->createBuilderConfig());
  MS_EXCEPTION_IF_NULL(config_);
  return true;
}

bool TrtConverterContext::Parser() {
  InitInputTable();
  InitValueNodeTable();

  std::vector<AnfNodePtr> node_list = TopoSort(func_graph_->get_return());
  const auto &converter_factory = TrtOpFactory::GetInstance();
  for (auto node : node_list) {
    if (!node->isa<CNode>()) {
      continue;
    }

    // Mark graph outputs
    std::string op_name = AnfAlgo::GetCNodePrimitive(node)->name();
    if (op_name == kReturnOpName) {
      std::vector<LayerInput> inputs;
      (void)LoadLayerInput(node, &inputs);

      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &input = inputs[i].tensor();
        std::string name = "return_output_" + std::to_string(i);
        input->setName(name.c_str());
        network_->markOutput(*input);
      }
      return true;
    }

    // Transform AnfNode To Trt layer.
    // Bypass control node including Depend, Load, UpdateState, TupleGetItem, MakeTuple.
    if (!AnfAlgo::IsRealKernel(node)) {
      continue;
    }

    ConvertFunc convert_func = converter_factory.GetConvertFunc(op_name);
    auto result = convert_func(node, this->shared_from_this());
    if (!result.first) {
      MS_LOG(ERROR) << op_name << " converter failed.";
      return false;
    }
    auto ret = StoreLayerOutput(node, result.second);
    if (!ret) {
      MS_LOG(ERROR) << op_name << " converter failed.";
      return false;
    }
  }

  MS_LOG(ERROR) << "Graph ended without return node.";
  return false;
}

bool TrtConverterContext::Serialize(std::string *model) {
  MS_EXCEPTION_IF_NULL(model);
  builder_->setMaxBatchSize(batch_size_);
  config_->setMaxWorkspaceSize(workspace_size_);
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
      weight.type = TrtUtils::MsDtypeToTrtDtype(tensor->data_type());
      weight.count = tensor->DataSize();
      output_map_[input_node][0] = LayerInput(weight);
    } else {
      nvinfer1::DataType trt_dtype = TrtUtils::MsDtypeToTrtDtype(AnfAlgo::GetOutputInferDataType(input_node, 0));
      nvinfer1::Dims trt_dims = TrtUtils::MsDimsToTrtDims(AnfAlgo::GetOutputInferShape(input_node, 0), false);
      nvinfer1::ITensor *tensor = network_->addInput(input->name().c_str(), trt_dtype, trt_dims);
      output_map_[input_node][0] = LayerInput(tensor);
    }
  }
  return true;
}

bool TrtConverterContext::InitValueNodeTable() {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph_);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  for (auto &value_node : kernel_graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);

    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
      std::vector<tensor::TensorPtr> tensors;
      TensorValueToTensor(node_value, &tensors);
      for (size_t i = 0; i < tensors.size(); i++) {
        const auto &tensor = tensors[i];
        nvinfer1::Weights weight;
        weight.values = tensor->data_c();
        weight.type = TrtUtils::MsDtypeToTrtDtype(tensor->data_type());
        weight.count = tensor->DataSize();
        output_map_[value_node][i] = LayerInput(weight);
      }
    }
  }
  return true;
}

bool TrtConverterContext::StoreLayerOutput(const AnfNodePtr &node, const std::vector<LayerInput> &nv_tensors) {
  if (nv_tensors.size() != AnfAlgo::GetOutputTensorNum(node)) {
    MS_LOG(INFO) << node->DebugString() << " output num not match. expect: " << AnfAlgo::GetOutputTensorNum(node)
                 << ", while got: " << nv_tensors.size();
  }

  for (size_t tensor_index = 0; tensor_index < nv_tensors.size(); ++tensor_index) {
    if (nv_tensors[tensor_index].tensor() != nullptr) {
      output_map_[node][tensor_index] = nv_tensors[tensor_index];

      std::ostringstream oss;
      nvinfer1::Dims dim = nv_tensors[tensor_index].tensor()->getDimensions();
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

bool TrtConverterContext::LoadLayerInput(const AnfNodePtr &node, std::vector<LayerInput> *inputs) {
  std::vector<session::KernelWithIndex> real_inputs;
  GetRealInputs(node, &real_inputs);
  for (auto item : real_inputs) {
    auto node_iter = output_map_.find(item.first);
    if (node_iter == output_map_.end()) {
      MS_LOG(ERROR) << "node: " << node->DebugString() << " not found.";
      return false;
    }

    auto out_iter = node_iter->second.find(item.second);
    if (out_iter == node_iter->second.end()) {
      MS_LOG(ERROR) << "node: " << node->DebugString() << "output index: " << item.second << " not found.";
      return false;
    }

    inputs->push_back(out_iter->second);
  }
  return true;
}

std::vector<AnfNodePtr> TrtConverterContext::GetGraphInputs() {
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

std::vector<session::KernelWithIndex> TrtConverterContext::GetGraphOutputs() {
  std::vector<session::KernelWithIndex> graph_outputs;
  GetRealInputs(func_graph_->get_return(), &graph_outputs);
  return graph_outputs;
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
