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

#define USE_DEPRECATED_API
#include "tools/converter/converter.h"
#include <memory>
#include <vector>
#include <set>
#include "tools/converter/converter_flags.h"
#include "src/common/log_adapter.h"
#include "tools/common/meta_graph_serializer.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "include/version.h"
#ifdef SUPPORT_TRAIN
#include "src/train/train_populate_parameter.h"
#endif
#include "include/registry/model_parser_registry.h"
#include "src/common/dynamic_library_loader.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/mindspore_importer.h"
#include "nnacl/op_base.h"
#include "tools/converter/micro/coder/coder.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "tools/common/tensor_util.h"
#include "include/api/model.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kMaxNum1024 = 1024;
void InitConverterParameters(const converter::Flags &flag, converter::ConverterParameters *converter_parameters) {
  MS_ASSERT(converter_parameters != nullptr);
  converter_parameters->fmk = flag.fmk;
  converter_parameters->model_file = flag.modelFile;
  converter_parameters->weight_file = flag.weightFile;
}
FuncGraphPtr ConvertGraph(const api::FuncGraphPtr &func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}
}  // namespace

FuncGraphPtr Converter::BuildFuncGraph(const converter::Flags &flag) {
  api::FuncGraphPtr func_graph_base = nullptr;
  if (flag.fmk == converter::FmkType::kFmkTypeMs) {
#ifdef SUPPORT_TRAIN
    kernel::PopulateTrainParameters();
#endif
    MindsporeImporter ms_import;
    func_graph_base = api::MakeShared<api::FuncGraph>(ms_import.ImportMindIR(flag));
  } else {
    model_parser_ = registry::ModelParserRegistry::GetModelParser(flag.fmk);
    if (model_parser_ == nullptr) {
      return nullptr;
    }
    converter::ConverterParameters converter_parameters;
    InitConverterParameters(flag, &converter_parameters);
    func_graph_base = model_parser_->Parse(converter_parameters);
  }
  if (func_graph_base == nullptr) {
    MS_LOG(ERROR) << "Get funcGraph failed for fmk: " << flag.fmkIn;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NOT_SUPPORT);
    return nullptr;
  }
  auto func_graph = ConvertGraph(func_graph_base);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is invalid.";
    return nullptr;
  }
  if (UpdateFuncGraphInputsAndOutputsDtype(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Update graph inputs and outputs dtype failed.";
    return nullptr;
  }
  return func_graph;
}

FuncGraphPtr Converter::BuildFuncGraph(const converter::Flags &flag, const void *buf, const size_t &size) {
  MindsporeImporter ms_import;
  FuncGraphPtr func_graph = ms_import.ImportMindIR(flag, buf, size);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Get funcGraph failed.";
    return nullptr;
  }

  if (UpdateFuncGraphInputsAndOutputsDtype(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Update graph inputs and outputs dtype failed.";
    return nullptr;
  }

  return func_graph;
}

schema::MetaGraphT *Converter::Convert(const std::unique_ptr<converter::Flags> &flag, const void *buf,
                                       const size_t &size) {
  if (flag == nullptr || buf == nullptr) {
    MS_LOG(ERROR) << "Input flag is nullptr";
    return nullptr;
  }
  auto graph = BuildFuncGraph(*flag, buf, size);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Parser/Import model return nullptr";
    return nullptr;
  }
  return TransferFuncGraph(flag, graph);
}

schema::MetaGraphT *Converter::Convert(const std::unique_ptr<converter::Flags> &flag) {
  if (flag == nullptr) {
    MS_LOG(ERROR) << "Input flag is nullptr";
    return nullptr;
  }

  // load plugin
  static std::vector<std::shared_ptr<DynamicLibraryLoader>> dl_loaders;
  if (!flag->pluginsPath.empty()) {
    for (auto &path : flag->pluginsPath) {
      auto dl_loader = std::make_shared<DynamicLibraryLoader>();
      MS_CHECK_TRUE_RET(dl_loader != nullptr, nullptr);
      auto status = dl_loader->Open(path);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "open dynamic library failed. " << path;
        return nullptr;
      }
      dl_loaders.emplace_back(dl_loader);
    }
  }

  auto graph = BuildFuncGraph(*flag);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Parser/Import model return nullptr";
    return nullptr;
  }

  return TransferFuncGraph(flag, graph);
}

schema::MetaGraphT *Converter::TransferFuncGraph(const std::unique_ptr<converter::Flags> &flag,
                                                 FuncGraphPtr func_graph) {
  MS_CHECK_TRUE_MSG(funcgraph_transform_ != nullptr, nullptr, "funcgraph_transform init failed");
  MS_CHECK_TRUE_MSG(metagraph_transform_ != nullptr, nullptr, "metagraph_transform_ init failed");

  // funcgraph compile
  func_graph = funcgraph_transform_->Transform(func_graph, flag.get());
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Transform anf graph return nullptr";
    return nullptr;
  }

  // protobuf -> flatbuffer
  auto meta_graph = Export(func_graph, false, false, flag->trainModel);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return nullptr;
  }

  // metagraph compile
  metagraph_transform_->SetGraphDef(meta_graph);
  auto status = metagraph_transform_->Transform(*flag);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform meta graph failed " << status;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    delete meta_graph;
    return nullptr;
  }

  status = UpdateGraphOutputName(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    delete meta_graph;
    return nullptr;
  }

  return meta_graph;
}

int CheckExistCustomOps(const schema::MetaGraphT *meta_graph, bool *exist_custom_nodes) {
  MS_CHECK_TRUE_MSG(meta_graph != nullptr && exist_custom_nodes != nullptr, RET_ERROR, "input params contain nullptr.");
  flatbuffers::FlatBufferBuilder fbb(kMaxNum1024);
  for (const auto &node : meta_graph->nodes) {
    auto prim = ConvertToPrimitive(node->primitive.get(), &fbb);
    if (prim == nullptr) {
      MS_LOG(ERROR) << "get primitive failed.";
      fbb.Clear();
      return RET_ERROR;
    }
    if (IsCustomNode(prim, static_cast<int>(SCHEMA_CUR))) {
      *exist_custom_nodes = true;
      break;
    }
  }
  fbb.Clear();
  return RET_OK;
}

int PreInference(const schema::MetaGraphT &meta_graph, const std::unique_ptr<converter::Flags> &flags) {
  if (flags->trainModel) {
    MS_LOG(WARNING) << "train model dont support pre-infer.";
    return RET_OK;
  }

  bool exist_custom_nodes = false;
  auto check_ret = CheckExistCustomOps(&meta_graph, &exist_custom_nodes);
  if (check_ret == RET_ERROR) {
    MS_LOG(ERROR) << "CheckExistCustomOps failed.";
    return RET_ERROR;
  }
  if (exist_custom_nodes) {
    MS_LOG(WARNING) << "exist custom nodes and will not be pre-infer.";
    return RET_OK;
  }
  mindspore::Model model;
  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
  auto offset = schema::MetaGraph::Pack(builder, &meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running ";
    std::cerr << "New context failed while running " << std::endl;
    return RET_ERROR;
  }
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  auto &device_list = context->MutableDeviceInfo();
  device_list.push_back(device_info);

  auto ret = model.Build(content, size, kMindIR, context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build error ";
    std::cerr << "Build error " << std::endl;
    return RET_ERROR;
  }
  for (auto &tensor : model.GetInputs()) {
    if (tensor.Shape().empty() || tensor.DataSize() <= 0 ||
        std::find(tensor.Shape().begin(), tensor.Shape().end(), -1) != tensor.Shape().end()) {
      MS_LOG(WARNING) << tensor.Name() << " is dynamic shape and will not be pre-infer.";
      return RET_OK;
    }
    auto status = GenerateRandomData(&tensor);
    if (status != RET_OK) {
      MS_LOG(ERROR) << tensor.Name() << "GenerateRandomData failed.";
      return status;
    }
  }
  std::vector<MSTensor> outputs;
  ret = model.Predict(model.GetInputs(), &outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Inference error ";
    std::cerr << "Inference error " << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

int RunConverter(int argc, const char **argv) {
  std::ostringstream oss;
  auto flags = std::make_unique<converter::Flags>();
  if (flags == nullptr) {
    oss.clear();
    oss << "NEW FLAGS ERROR:" << RET_MEMORY_FAILED << " " << GetErrorInfo(RET_MEMORY_FAILED);
    MS_LOG(ERROR) << oss.str();
    std::cout << oss.str() << std::endl;
    return RET_MEMORY_FAILED;
  }
  auto status = flags->Init(argc, argv);
  if (status != RET_OK) {
    if (status != RET_SUCCESS_EXIT) {
      oss.clear();
      oss << "CONVERTER::FLAGS INIT FAILED:" << status << " " << GetErrorInfo(status);
      MS_LOG(ERROR) << oss.str();
      std::cout << oss.str() << std::endl;
    }
    return status;
  }
  // Load graph
  MS_LOG(DEBUG) << "start reading model file";
  Converter cvt;
  auto meta_graph = cvt.Convert(flags);
  NotSupportOp::GetInstance()->PrintOps();
  status = ReturnCode::GetSingleReturnCode()->status_code();
  if (meta_graph == nullptr) {
    oss.clear();
    oss << "CONVERT RESULT FAILED:" << status << " " << GetErrorInfo(status);
    MS_LOG(ERROR) << oss.str();
    std::cout << oss.str() << std::endl;
    status = RET_ERROR;
    return status;
  }
  //   save graph to file
  meta_graph->version = Version();

  if (flags->infer) {
    status = PreInference(*meta_graph, flags);
    if (status != RET_OK) {
      oss.clear();
      oss << "PRE INFERENCE FAILED:" << status << " " << GetErrorInfo(status);
      MS_LOG(ERROR) << oss.str();
      std::cout << oss.str() << std::endl;
      delete meta_graph;
      return status;
    }
  }

  if (flags->microParam.enable_micro) {
    status = micro::Coder::MicroSourceCodeGeneration(*meta_graph, flags->outputFile, flags->microParam.codegen_mode,
                                                     flags->microParam.target, flags->microParam.support_parallel,
                                                     flags->microParam.debug_mode);
    if (status != RET_OK) {
      delete meta_graph;
      oss.clear();
      oss << "MICRO CODEGEN FAILED:" << status << " " << GetErrorInfo(status);
      MS_LOG(ERROR) << oss.str();
      std::cout << oss.str() << std::endl;
      return status;
    }
  } else {
    status = MetaGraphSerializer::Save(*meta_graph, flags->outputFile, flags->encKey, flags->keyLen, flags->encMode);
    if (status != RET_OK) {
      delete meta_graph;
      oss.clear();
      oss << "SAVE GRAPH FAILED:" << status << " " << GetErrorInfo(status);
      MS_LOG(ERROR) << oss.str();
      std::cout << oss.str() << std::endl;
      return status;
    }
  }
  // clear key
  status = memset_s(flags->encKey, converter::kEncMaxLen, 0, converter::kEncMaxLen);
  if (status != EOK) {
    MS_LOG(ERROR) << "memset failed.";
    return RET_ERROR;
  }
  delete meta_graph;
  oss.clear();
  oss << "CONVERT RESULT SUCCESS:" << status;
  MS_LOG(INFO) << oss.str();
  std::cout << oss.str() << std::endl;
  return status;
}
}  // namespace lite
}  // namespace mindspore
