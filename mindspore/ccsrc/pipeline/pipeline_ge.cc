/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "pipeline/pipeline_ge.h"

#include <sstream>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>

#include "debug/anf_ir_dump.h"
#include "ir/meta_tensor.h"
#include "transform/convert.h"
#include "transform/df_graph_manager.h"
#include "transform/graph_builder.h"
#include "transform/graph_runner.h"
#include "debug/draw.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using TensorOrderMap = std::map<std::string, std::shared_ptr<Tensor>>;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::transform::DfGraphConvertor;
using mindspore::transform::DfGraphManager;
using mindspore::transform::GeTensorPtr;
using mindspore::transform::MeTensorPtr;
using mindspore::transform::Status;
using mindspore::transform::TransformUtil;

void DoExecNonInputGraph(const std::string &phase) {
  std::vector<GeTensorPtr> ge_tensors;
  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;
  run_options.name = phase;
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();

  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return;
  }
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph:" << run_options.name << " failed";
      return;
    }
  }
}

void SetGeOption(const std::map<std::string, std::string> &options) {
  ConfigManager::GetInstance().set_ge_initialize_options(options);
}

Status CreateSessionAndGraphRunner(bool is_training = true) {
  std::shared_ptr<ge::Session> sess = DfGraphManager::GetInstance().GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    if (is_training) {
      options["ge.trainFlag"] = "1";
      options["ge.streamNum"] = "100";
      options["ge.enabledLocalFmkop"] = "1";
      options["ge.hcomParallel"] = "1";
    } else {
      options["ge.trainFlag"] = "0";
    }

    options["ge.enablePrintOpPass"] = "0";
    sess = transform::GraphRunner::NewSession(options);
    if (sess == nullptr) {
      MS_LOG(ERROR) << "Init data graph failed, because of create Ge session failed";
      return Status::FAILED;
    } else {
      DfGraphManager::GetInstance().SetGeSession(sess);
    }
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = std::make_shared<transform::GraphRunner>(options);
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Create new graph runner failed";
    return Status::FAILED;
  } else {
    DfGraphManager::GetInstance().SetGraphRunner(graph_runner);
  }

  return Status::SUCCESS;
}

bool InitExecDatasetGe(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, const std::string &phase) {
  std::vector<int64_t> ge_types;
  (void)std::transform(types.begin(), types.end(), std::back_inserter(ge_types), [](const TypePtr &i) -> int64_t {
    return transform::TransformUtil::ConvertDataType(i->type_id());
  });

  ConfigManager::GetInstance().set_dataset_mode(DatasetMode::DS_SINK_MODE);
  ConfigManager::GetInstance().set_iter_num(size);
  ConfigManager::GetInstance().set_dataset_phase(phase);

  DatasetGraphParam param(queue_name, size, batch_size, ge_types, shapes, input_indexes);
  ConfigManager::GetInstance().set_dataset_param(param);

  if (transform::BuildDatasetGraph(param, phase) != transform::SUCCESS) {
    MS_LOG(ERROR) << "Build dateset graph failed.";
    return false;
  }

#if ENABLE_TRAIN
  (void)setenv("GE_TRAIN", "1", 1);
#else
  (void)setenv("GE_TRAIN", "0", 1);
#endif

  if (CreateSessionAndGraphRunner(static_cast<bool>(ENABLE_TRAIN)) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return false;
  }

  MS_LOG(INFO) << "DoExecNonInputGraph:" << phase;
  DoExecNonInputGraph(phase);

  return true;
}

void ConvertObjectToTensors(const py::dict &dict, TensorOrderMap *const tensors) {
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);
    if (py::isinstance<py::float_>(item.second.attr("default_input"))) {
      // convert float to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeFloat32, std::vector<int>({1}));
      *(static_cast<float *>(tensor->data_c(true))) = py::cast<float>(item.second.attr("default_input"));
    } else if (py::isinstance<py::int_>(item.second.attr("default_input"))) {
      // convert int to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeInt32, std::vector<int>({1}));
      *(static_cast<float *>(tensor->data_c(true))) = py::cast<float>(item.second.attr("default_input"));
    } else if (py::hasattr(item.second.attr("default_input"), PYTHON_TENSOR_FLAG)) {
      // cast tensor
      tensor = py::cast<std::shared_ptr<Tensor>>(item.second.attr("default_input"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

bool AddDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::dict &init_params,
                const std::string &phase, const py::object &broadcast_params) {
  FuncGraphPtr anf_graph = info.at(phase)->func_graph;
  DfGraphConvertor convertor(anf_graph);

  size_t pos = phase.find('.');
  std::string net_id = ((pos == std::string::npos || pos == phase.size() - 1) ? phase : phase.substr(pos + 1));
  std::string phase_prefix = phase.substr(0, pos);

  if (phase_prefix == "export") {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    convertor.set_training(false);
  }

  TensorOrderMap init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors);
  (void)convertor.ConvertAllNode().InitParam(init_tensors).BuildGraph();

  if (broadcast_params != py::none()) {
    if (!py::isinstance<py::dict>(broadcast_params)) {
      MS_LOG(ERROR) << "Invalid broadcast params, it must be py::dict type";
      return false;
    }
    py::dict broadcast = broadcast_params.cast<py::dict>();
    if (broadcast.empty()) {
      (void)convertor.GenerateBroadcastGraph(init_tensors);
    } else {
      TensorOrderMap broadcast_tensors{};
      ConvertObjectToTensors(broadcast, &broadcast_tensors);
      (void)convertor.GenerateBroadcastGraph(broadcast_tensors);
    }
    MS_LOG(INFO) << "Generate broadcast graph with params and broadcast_empty is " << broadcast.empty();
  }

  (void)convertor.GenerateCheckpointGraph();
  if (convertor.ErrCode() != 0) {
    DfGraphManager::GetInstance().ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << convertor.ErrCode();
    return false;
  }

  if (MsContext::GetInstance()->save_graphs_flag()) {
    convertor.DrawComputeGraph(GetFilePathName("ge_graph.dot"));                      // for debug
    convertor.DrawInitGraph(GetFilePathName("init_graph.dot"));                       // for debug
    convertor.DrawSaveCheckpointGraph(GetFilePathName("save_checkpoint_graph.dot"));  // for debug
  }
  std::string init_graph = "init_subgraph." + net_id;
  std::string checkpoint_name = "save." + net_id;
  if (phase.find("train") != std::string::npos) {
    (void)DfGraphManager::GetInstance().AddGraph(phase, convertor.GetComputeGraph(), {{"ge.exec.variable_acc", "1"}});
  } else {
    (void)DfGraphManager::GetInstance().AddGraph(phase, convertor.GetComputeGraph());
  }
  (void)DfGraphManager::GetInstance().AddGraph(init_graph, convertor.GetInitGraph());
  (void)DfGraphManager::GetInstance().AddGraph(BROADCAST_GRAPH_NAME, convertor.GetBroadcastGraph());

  Status ret = DfGraphManager::GetInstance().AddGraph(checkpoint_name, convertor.GetSaveCheckpointGraph());
  if (ret == Status::SUCCESS) {
    DfGraphManager::GetInstance().SetAnfGraph(checkpoint_name, anf_graph);
  }

  return true;
}

FuncGraphPtr BuildDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::dict &init_params,
                          const std::string &phase, const py::object &broadcast_params) {
  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }
  FuncGraphPtr anf_graph = info.at(phase)->func_graph;

  if (MsContext::GetInstance()->save_graphs_flag()) {
    draw::Draw(GetFilePathName("anf_graph.dot"), anf_graph);  // for debug
    DumpIR(GetFilePathName("anf_graph.ir"), anf_graph, true);
  }

  if (!AddDFGraph(info, init_params, phase, broadcast_params)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

#if ENABLE_TRAIN
  (void)setenv("GE_TRAIN", "1", 1);
#else
  (void)setenv("GE_TRAIN", "0", 1);
#endif

  if (CreateSessionAndGraphRunner(static_cast<bool>(ENABLE_TRAIN)) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return nullptr;
  }

  return anf_graph;
}

void RunGEInitGraph(const py::dict &init_params, const std::string &phase) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  TensorOrderMap inputs_with_name{};
  ConvertObjectToTensors(init_params, &inputs_with_name);
  std::vector<tensor::TensorPtr> inputs;
  (void)std::transform(inputs_with_name.begin(), inputs_with_name.end(), std::back_inserter(inputs),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });

  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "Args convert to ge tensor error.";
    return;
  }
  MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size() << ".";

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = phase;
  if (DfGraphManager::GetInstance().GetGraphByName(phase) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << phase << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << phase << " graph failed.";
    }

    MS_LOG(INFO) << "Exec " << phase << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (DfGraphManager::GetInstance().GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
      if (ret != Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(INFO) << "Exec broadcast graph success.";
    }
  }
}

py::object ExtractGeneralCnodeRet(const AbstractBasePtr &cnode_data, const py::tuple &data, size_t *count) {
  MS_EXCEPTION_IF_NULL(cnode_data);
  if (*count >= data.size()) {
    MS_LOG(EXCEPTION) << "The number of elements in the outputs : " << data.size()
                      << " less than the number of elements required. ";
  }

  if (cnode_data->isa<AbstractTensor>()) {
    BaseShapePtr shape = cnode_data->BuildShape();
    auto shape_act = shape->cast<abstract::ShapePtr>()->shape();
    Tensor tensor_exp = py::cast<Tensor>(data[*count]);
    if (shape_act != tensor_exp.shape()) {
      MS_LOG(EXCEPTION) << "The shape of the tensor returned from GE is not the same as "
                           "the shape of the tensor derived from ME.";
    }
    return data[(*count)++];
  }

  if (!cnode_data->isa<AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "The output of operator in the final anf graph could "
                      << "only be a tensor or a tuple of tensor, but got " << cnode_data->BuildValue()->ToString()
                      << ".";
  }
  auto data_tp = cnode_data->cast<AbstractTuplePtr>();
  auto elements = data_tp->elements();
  size_t size = data_tp->size();
  py::tuple tp = py::tuple(size);
  for (size_t i = 0; i < size; i++) {
    tp[i] = ExtractGeneralCnodeRet(elements[i], data, count);
  }
  return std::move(tp);
}

py::object StructureOutput(const AnfNodePtr &output_node, const py::tuple &data, size_t *count) {
  MS_EXCEPTION_IF_NULL(output_node);

  if (output_node->isa<ValueNode>()) {
    return ValuePtrToPyData(GetValueNode(output_node));
  }

  if (*count >= data.size()) {
    MS_LOG(EXCEPTION) << "The number of elements in the outputs : " << data.size()
                      << " less than the number of elements required. ";
  }
  if (output_node->isa<Parameter>()) {
    return data[(*count)++];
  }

  auto output_c = output_node->cast<CNodePtr>();
  if (output_c == nullptr) {
    MS_LOG(EXCEPTION) << "The final anf graph could only have constant, parameter, and operator, but got "
                      << output_node->ToString();
  }

  if (output_c->IsApply(prim::kPrimMakeTuple)) {
    auto input_list = output_c->inputs();
    size_t size = input_list.size();
    py::tuple tp = py::tuple(size - 1);
    for (size_t i = 1; i < size; i++) {
      tp[i - 1] = StructureOutput(input_list[i], data, count);
    }
    return std::move(tp);
  }
  if (output_c->IsApply(prim::kPrimDepend)) {
    return StructureOutput(output_c->input(1), data, count);
  }

  return ExtractGeneralCnodeRet(output_c->abstract(), data, count);
}

std::shared_ptr<py::object> DoExecGraph(const FuncGraphPtr &graph, const std::vector<MeTensorPtr> &inputs,
                                        const std::string &phase) {
  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Convert me args to ge tensor error.";
  }

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = phase;

  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();

  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph failed";
      return nullptr;
    }
  }

  std::vector<MeTensorPtr> me_outputs = TransformUtil::ConvertGeTensors(ge_outputs);
  if (me_outputs.size() != ge_outputs.size()) {
    MS_LOG(WARNING) << "Convert output Ge tensor to Me tensor failed";
  }

  py::tuple outputs(me_outputs.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    outputs[i] = *me_outputs[i];
  }

  std::shared_ptr<py::object> ret = nullptr;

  AnfNodePtr output_node = graph->get_return()->input(1);
  MS_EXCEPTION_IF_NULL(output_node);
  size_t count = 0;
  py::object oj = StructureOutput(output_node, outputs, &count);
  ret = std::make_shared<py::object>(oj);

  return ret;
}

void ProcessGeArg(const std::map<std::string, ExecutorInfoPtr> &info, const py::tuple &args, const std::string &phase,
                  std::vector<tensor::TensorPtr> *inputs) {
  // check the arg and use the ExecutorPy args
  std::size_t size = args.size();

  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }

  auto arg_size = info.at(phase)->arg_list_size;
  if (size != arg_size) {
    MS_LOG(EXCEPTION) << "The real arg num : size = " << size << ". graph_arg_size = " << arg_size;
  }

  // process the first args of tensor
  // only in dataset normal(non-sink) mode, fp_bp graph need input tensors
  if (ConfigManager::GetInstance().dataset_mode() == DS_NORMAL_MODE) {
    for (std::size_t i = 0; i < size; i++) {
      ValuePtr converted = nullptr;
      bool succ = parse::ConvertData(args[i], &converted);
      if (!succ) {
        MS_LOG(EXCEPTION) << "Args convert error";
      }
      if (converted->isa<tensor::Tensor>()) {
        (*inputs).push_back(converted->cast<tensor::TensorPtr>());
      } else {
        MS_LOG(EXCEPTION) << "Args " << converted->ToString() << " is not tensor";
      }
    }
  }
}

py::object ExecDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::tuple &args,
                       const std::string &phase) {
  std::string phase_prefix = GetPhasePrefix(phase);

  if (phase_prefix == "save") {
    DoExecNonInputGraph(phase);
    ConfigManager::GetInstance().ResetConfig();
    return py::none();
  }

  if (info.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "There is no phase:" << phase;
  }

  FuncGraphPtr anf_graph = info.at(phase)->func_graph;

#ifdef ENABLE_INFER
  // Now don't use the graph because the exec ge function don't take effect
  MS_EXCEPTION_IF_NULL(info.at(phase)->func_graph);
  if (ENABLE_TRAIN != info.at(phase)->func_graph->flags()["training"]) {
    MS_LOG(ERROR) << "Graph training mode mismatch mode of libraries";
    ConfigManager::GetInstance().ResetConfig();
    return py::none();
  }
#endif

  std::shared_ptr<py::object> ret_val = std::make_shared<py::object>();
  // We will not execute graph when output is constant or just input itself.
  if (IsGraphOutputValueNodeOrParameter(info.at(phase)->func_graph->output(), args, ret_val)) {
    ConfigManager::GetInstance().ResetConfig();
    return *ret_val;
  }

  std::vector<tensor::TensorPtr> inputs;
  ProcessGeArg(info, args, phase, &inputs);

  std::shared_ptr<py::object> ret = DoExecGraph(anf_graph, inputs, phase);
  ConfigManager::GetInstance().ResetConfig();
  if (ret != nullptr) {
    return *ret;
  } else {
    MS_LOG(EXCEPTION) << "Exec graph failed";
  }
}
void ExportDFGraph(const std::string &file_name, const std::string &phase) {
  MS_LOG(DEBUG) << "ExportGraph Begin";
  transform::DfGraphWrapperPtr wrap_ptr = DfGraphManager::GetInstance().GetGraphByName(phase);
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return;
  }

  transform::DfGraphPtr ge_graph = wrap_ptr->graph_ptr_;
  if (nullptr == ge_graph) {
    MS_LOG(ERROR) << "The export graph is null";
    return;
  }

  (void)ge_graph->SaveToFile(file_name);

  MS_LOG(DEBUG) << "ExportGraph End";
}
}  // namespace pipeline
}  // namespace mindspore
