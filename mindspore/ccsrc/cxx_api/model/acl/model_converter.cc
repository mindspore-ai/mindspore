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

#include "cxx_api/model/acl/model_converter.h"
#include <memory>
#include "pybind11/pybind11.h"
#include "transform/graph_ir/convert.h"
#include "transform/graph_ir/graph_runner.h"
#include "core/load_mindir/load_model.h"
#include "mindspore/core/utils/ms_context.h"
#include "backend/kernel_compiler/oplib/oplib.h"

#include "graph/model.h"
#include "cxx_api/model/model_converter_utils/multi_process.h"

namespace py = pybind11;

namespace mindspore::api {
namespace {
transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

bool CreateSessionAndGraphRunner() {
  std::shared_ptr<ge::Session> sess = transform::DfGraphManager::GetInstance().GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    options["ge.trainFlag"] = "0";
    options["ge.enablePrintOpPass"] = "0";
    sess = transform::GraphRunner::NewSession(options);
    if (sess == nullptr) {
      MS_LOG(ERROR) << "Init data graph failed, because of create Ge session failed";
      return false;
    } else {
      transform::DfGraphManager::GetInstance().SetGeSession(sess);
    }
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = std::make_shared<transform::GraphRunner>(options);
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Create new graph runner failed";
    return false;
  } else {
    transform::DfGraphManager::GetInstance().SetGraphRunner(graph_runner);
  }

  return true;
}

}  // namespace

std::shared_ptr<FuncGraph> ModelConverter::ConvertMindIrToFuncGraph(const Buffer &model_data) {
  try {
    auto anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(model_data.Data()), model_data.DataSize());
    return anf_graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Load MindIR failed.";
    return nullptr;
  }
}

transform::DfGraphPtr ModelConverter::ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) {
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    // normalize name
    std::string name = para->name();
    for (auto pos = name.find(':'); pos != std::string::npos; pos = name.find(':')) {
      name = name.substr(0, pos) + "_" + name.substr(pos + 1);
      MS_LOG(INFO) << name;
    }
    para->set_name(name);
  }

  transform::DfGraphConvertor convertor(anf_graph);
  std::string net_id = "0";
  std::string init_graph = "init_subgraph." + net_id;
  std::string checkpoint_name = "save." + net_id;

  convertor.set_training(false);
  (void)convertor.ConvertAllNode().InitParam(GetParams(anf_graph)).BuildGraph();
  (void)convertor.GenerateCheckpointGraph();
  if (convertor.ErrCode() != 0) {
    transform::DfGraphManager::GetInstance().ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << convertor.ErrCode();
    return nullptr;
  }
  (void)transform::DfGraphManager::GetInstance().AddGraph(anf_graph->ToString(), convertor.GetComputeGraph());
  (void)transform::DfGraphManager::GetInstance().AddGraph(init_graph, convertor.GetInitGraph());
  (void)transform::DfGraphManager::GetInstance().AddGraph(BROADCAST_GRAPH_NAME, convertor.GetBroadcastGraph());

  transform::Status ret =
    transform::DfGraphManager::GetInstance().AddGraph(checkpoint_name, convertor.GetSaveCheckpointGraph());
  if (ret == transform::Status::SUCCESS) {
    transform::DfGraphManager::GetInstance().SetAnfGraph(checkpoint_name, anf_graph);
  }

  (void)setenv("GE_TRAIN", "0", 1);

  if (!CreateSessionAndGraphRunner()) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return nullptr;
  }

  auto wrap_ptr = transform::DfGraphManager::GetInstance().GetGraphByName(anf_graph->ToString());
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return nullptr;
  }
  transform::DfGraphPtr &ge_graph = wrap_ptr->graph_ptr_;
  if (ge_graph == nullptr) {
    MS_LOG(ERROR) << "The export graph is null";
    return nullptr;
  }

  return ge_graph;
}

Buffer ModelConverter::BuildAirModel(const transform::DfGraphPtr &graph,
                                     const std::map<std::string, std::string> &acl_options) {
  ge::ModelBufferData model;
  auto ge_options = acl_options;
  ge_options.emplace(ge::ir_option::SOC_VERSION, "Ascend310");
  auto ret = ge::aclgrphBuildInitialize(ge_options);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildInitialize fail.";
    return Buffer();
  }

  ret = ge::aclgrphBuildModel(*graph, acl_options, model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildModel fail.";
    return Buffer();
  }

  ge::aclgrphBuildFinalize();
  return Buffer(model.data.get(), model.length);
}

void ModelConverter::RegAllOp() {
  static std::mutex init_mutex;
  static bool Initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  Initialized = true;
  MsContext::GetInstance()->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  Py_Initialize();
  auto c_expression = PyImport_ImportModule("mindspore._c_expression");
  MS_EXCEPTION_IF_NULL(c_expression);
  PyObject *c_expression_dict = PyModule_GetDict(c_expression);
  MS_EXCEPTION_IF_NULL(c_expression_dict);

  PyObject *op_info_loader_class = PyDict_GetItemString(c_expression_dict, "OpInfoLoaderPy");
  MS_EXCEPTION_IF_NULL(op_info_loader_class);
  PyObject *op_info_loader = PyInstanceMethod_New(op_info_loader_class);
  MS_EXCEPTION_IF_NULL(op_info_loader);
  PyObject *op_info_loader_ins = PyObject_CallObject(op_info_loader, nullptr);
  MS_EXCEPTION_IF_NULL(op_info_loader_ins);
  auto all_ops_info_vector_addr_ul = PyObject_CallMethod(op_info_loader_ins, "get_all_ops_info", nullptr);
  MS_EXCEPTION_IF_NULL(all_ops_info_vector_addr_ul);
  auto all_ops_info_vector_addr = PyLong_AsVoidPtr(all_ops_info_vector_addr_ul);
  auto all_ops_info = static_cast<std::vector<kernel::OpInfo *> *>(all_ops_info_vector_addr);
  for (auto op_info : *all_ops_info) {
    kernel::OpLib::RegOpInfo(std::shared_ptr<kernel::OpInfo>(op_info));
  }
  all_ops_info->clear();
  delete all_ops_info;
  Py_DECREF(op_info_loader);
  Py_DECREF(op_info_loader_class);
  Py_DECREF(c_expression_dict);
  Py_DECREF(c_expression);
}

Buffer ModelConverter::ReadFile(const std::string &file) {
  Buffer buffer;
  if (file.empty()) {
    MS_LOG(ERROR) << "Pointer file is nullptr";
    return buffer;
  }
  std::string realPath = file;
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << realPath << " is not exist";
    return buffer;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << realPath << "open failed";
    return buffer;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  buffer.ResizeData(size);
  if (buffer.DataSize() != size) {
    MS_LOG(ERROR) << "Malloc buf failed, file: " << realPath;
    ifs.close();
    return buffer;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

Buffer ModelConverter::LoadMindIR(const Buffer &model_data) {
  if (Py_IsInitialized() == 0) {
    MS_LOG_INFO << "Call LoadMindIRInner directly";
    return LoadMindIRInner(model_data);
  }
  MultiProcess multi_process;
  Buffer buffer_ret;
  auto parent_process = [&model_data, &buffer_ret](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    // send original model to child
    auto status = multi_process->SendMsg(model_data.Data(), model_data.DataSize());
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Send original model to child process failed";
      return FAILED;
    }
    // receive convert model result from child
    CreateBufferCall call = [&buffer_ret](size_t msg_len) -> uint8_t * {
      buffer_ret.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(buffer_ret.MutableData());
    };
    status = multi_process->ReceiveMsg(call);
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Receive result model from child process failed";
      return FAILED;
    }
    return SUCCESS;
  };
  auto child_process = [this](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    // receive original model from parent
    Buffer model;
    CreateBufferCall call = [&model](size_t msg_len) -> uint8_t * {
      model.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(model.MutableData());
    };
    auto status = multi_process->ReceiveMsg(call);
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Receive original model from parent process failed";
      return FAILED;
    }
    Buffer model_result = LoadMindIRInner(model);
    if (model_result.DataSize() == 0) {
      MS_LOG_ERROR << "Convert model from MindIR to OM failed";
      return FAILED;
    }
    // send result model to parent
    status = multi_process->SendMsg(model_result.Data(), model_result.DataSize());
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Send result model to parent process failed";
      return FAILED;
    }
    return SUCCESS;
  };
  auto status = multi_process.MainProcess(parent_process, child_process);
  if (!status.IsSuccess()) {
    MS_LOG_ERROR << "Convert MindIR model to OM model failed";
  } else {
    MS_LOG_INFO << "Convert MindIR model to OM model success";
  }
  return buffer_ret;
}

Buffer ModelConverter::LoadAscendIR(const Buffer &model_data) {
  if (Py_IsInitialized() == 0) {
    MS_LOG_INFO << "Call LoadAscendIRInner directly";
    return LoadAscendIRInner(model_data);
  }
  MultiProcess multi_process;
  Buffer buffer_ret;
  auto parent_process = [&model_data, &buffer_ret](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    // send original model to child
    auto status = multi_process->SendMsg(model_data.Data(), model_data.DataSize());
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Send original model to child process failed";
      return FAILED;
    }
    // receive convert model result from child
    CreateBufferCall call = [&buffer_ret](size_t msg_len) -> uint8_t * {
      buffer_ret.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(buffer_ret.MutableData());
    };
    status = multi_process->ReceiveMsg(call);
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Receive result model from child process failed";
      return FAILED;
    }
    return SUCCESS;
  };
  auto child_process = [this](MultiProcess *multi_process) -> Status {
    MS_EXCEPTION_IF_NULL(multi_process);
    // receive original model from parent
    Buffer model;
    CreateBufferCall call = [&model](size_t msg_len) -> uint8_t * {
      model.ResizeData(msg_len);
      return reinterpret_cast<uint8_t *>(model.MutableData());
    };
    auto status = multi_process->ReceiveMsg(call);
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Receive original model from parent process failed";
      return FAILED;
    }
    Buffer model_result = LoadAscendIRInner(model);
    if (model_result.DataSize() == 0) {
      MS_LOG_ERROR << "Convert model from AIR to OM failed";
      return FAILED;
    }
    // send result model to parent
    status = multi_process->SendMsg(model_result.Data(), model_result.DataSize());
    if (!status.IsSuccess()) {
      MS_LOG_ERROR << "Send result model to parent process failed";
      return FAILED;
    }
    return SUCCESS;
  };
  auto status = multi_process.MainProcess(parent_process, child_process);
  if (!status.IsSuccess()) {
    MS_LOG_ERROR << "Convert AIR model to OM model failed";
  } else {
    MS_LOG_INFO << "Convert AIR model to OM model success";
  }
  return buffer_ret;
}

Buffer ModelConverter::LoadMindIRInner(const Buffer &model_data) {
  RegAllOp();
  Py_Initialize();
  auto func_graph = ConvertMindIrToFuncGraph(model_data);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Convert MindIR to FuncGraph failed.";
    return Buffer();
  }

  auto df_graph = ConvertFuncGraphToAIR(func_graph);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return Buffer();
  }

  std::map<std::string, std::string> acl_options;
  if (options_ != nullptr) {
    acl_options = options_->GenAclOptions();
  }

  auto om_data = BuildAirModel(df_graph, acl_options);
  return om_data;
}

Buffer ModelConverter::LoadAscendIRInner(const Buffer &model_data) {
  RegAllOp();
  ge::Model load_model = ge::Model("loadmodel", "version2");
  ge::Status ret =
    ge::Model::Load(reinterpret_cast<const uint8_t *>(model_data.Data()), model_data.DataSize(), load_model);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Load AscendIR failed, ret = " << ret;
    return Buffer();
  }

  transform::DfGraphPtr df_graph = std::make_shared<transform::DfGraph>(load_model.GetGraph());
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return Buffer();
  }

  std::map<std::string, std::string> acl_options;
  if (options_ != nullptr) {
    acl_options = options_->GenAclOptions();
  }

  auto om_data = BuildAirModel(df_graph, acl_options);
  return om_data;
}
}  // namespace mindspore::api
