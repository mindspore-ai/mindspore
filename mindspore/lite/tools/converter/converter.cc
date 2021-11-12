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
namespace mindspore {
namespace lite {
namespace {
void InitConverterParameters(const converter::Flags &flag, converter::ConverterParameters *converter_parameters) {
  MS_ASSERT(converter_parameters != nullptr);
  converter_parameters->fmk = flag.fmk;
  converter_parameters->model_file = flag.modelFile;
  converter_parameters->weight_file = flag.weightFile;
}
}  // namespace

FuncGraphPtr Converter::BuildFuncGraph(const converter::Flags &flag) {
  api::FuncGraphPtr func_graph_base = nullptr;
  if (flag.fmk == converter::FmkType::kFmkTypeMs) {
#ifdef SUPPORT_TRAIN
    kernel::PopulateTrainParameters();
#endif
    MindsporeImporter ms_import;
    func_graph_base = ms_import.ImportMindIR(flag);
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
  auto func_graph = std::dynamic_pointer_cast<FuncGraph>(func_graph_base);
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
  status = MetaGraphSerializer::Save(*meta_graph, flags->outputFile);
  if (status != RET_OK) {
    oss.clear();
    oss << "SAVE GRAPH FAILED:" << status << " " << GetErrorInfo(status);
    MS_LOG(ERROR) << oss.str();
    std::cout << oss.str() << std::endl;
    return status;
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
