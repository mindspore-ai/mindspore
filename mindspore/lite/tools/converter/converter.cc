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
#include "tools/common/storage.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "include/version.h"
#include "src/train/train_populate_parameter.h"
#include "include/registry/model_parser_registry.h"
#include "src/common/dynamic_library_loader.h"
#include "tools/converter/export_model.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/mindspore_importer.h"
namespace mindspore {
namespace lite {
FuncGraphPtr Converter::BuildFuncGraph(const converter::Flags &flag) {
  FuncGraphPtr func_graph = nullptr;
  if (flag.fmk == converter::FmkType::FmkType_MS) {
    kernel::PopulateTrainParameters();
    MindsporeImporter ms_import;
    func_graph = ms_import.ImportMindIR(flag);
    if (func_graph == nullptr) {
      return nullptr;
    }
  } else {
    model_parser_ = ModelParserRegistry::GetInstance()->GetModelParser(flag.fmk);
    if (model_parser_ == nullptr) {
      return nullptr;
    }
    func_graph = model_parser_->Parse(flag);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Get funcGraph failed for fmk: " << flag.fmkIn;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NOT_SUPPORT);
    return nullptr;
  }
  if (UpdateFuncGraphInputsAndOutputsDtype(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Update graph inputs and outputs dtype failed.";
    return nullptr;
  }
  return func_graph;
}

schema::MetaGraphT *Converter::Convert(const std::unique_ptr<converter::Flags> &flag) {
  if (flag == nullptr) {
    MS_LOG(ERROR) << "Input flag is nullptr";
    return nullptr;
  }

  // load plugin
  if (!flag->pluginsPath.empty()) {
    DynamicLibraryLoader dynamic_library_loader{};
    for (auto &path : flag->pluginsPath) {
      auto status = dynamic_library_loader.Open(path.c_str());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "open dynamic library failed.";
        return nullptr;
      }
      dynamic_library_loader.Close();
    }
  }

  auto graph = BuildFuncGraph(*flag);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Parser/Import model return nullptr";
    return nullptr;
  }

  // funcgraph compile
  graph = funcgraph_transform_->Transform(graph, flag.get());
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Transform anf graph return nullptr";
    return nullptr;
  }

  // protobuf -> flatbuffer
  auto meta_graph = Export(graph, false, false, flag->trainModel);
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
  // Init dump graph func
  ExportModelInit(flags.get());
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
  status = Storage::Save(*meta_graph, flags->outputFile);
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
