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
#include "tools/converter/registry/model_parser_registry.h"
#include "src/common/dynamic_library_loader.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/fusion/squeeze_fusion.h"
#include "tools/optimizer/graph/conv1d_weight_expanding_pass.h"
#include "tools/optimizer/graph/primitive_adjust_pass.h"
#include "tools/optimizer/graph/mindir_adjust_pass.h"

namespace mindspore {
namespace lite {
STATUS Converter::AdjustForMindir(const FuncGraphPtr &func_graph, const converter::Flags &flag) {
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  // adjust for mindir
  for (auto fg : all_func_graphs) {
    auto primitive_adjust_pass = std::make_shared<opt::PrimitiveAdjustPass>();
    primitive_adjust_pass->SetFmkType(flag.fmk);
    if (!primitive_adjust_pass->Run(fg)) {
      MS_LOG(ERROR) << "primitive adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    auto mindir_adjust_pass = std::make_shared<opt::MindirAdjustPass>();
    mindir_adjust_pass->SetFmkType(flag.fmk);
    mindir_adjust_pass->SetQuantType(flag.quantType);
    mindir_adjust_pass->SetTrainFlag(flag.trainModel);
    if (!mindir_adjust_pass->Run(fg)) {
      MS_LOG(ERROR) << "mindir adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

FuncGraphPtr Converter::BuildFuncGraph(const converter::Flags &flag) {
  FuncGraphPtr func_graph = nullptr;
  if (flag.fmk == converter::FmkType::FmkType_MS) {
    kernel::PopulateTrainParameters();
    func_graph = LoadMindIR(flag.modelFile);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "get funcGraph failed for fmk:MINDIR";
      return nullptr;
    }
    func_graph->set_attr("graph_name", MakeValue("main_graph"));
    func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_MS)));
    if (AdjustForMindir(func_graph, flag) != RET_OK) {
      MS_LOG(ERROR) << "AdjustForMindir failed.";
      return nullptr;
    }
  } else {
    model_parser_ = ModelParserRegistry::GetInstance()->GetModelParser(flag.fmkIn);
    if (model_parser_ == nullptr) {
      MS_LOG(ERROR) << "get funcGraph failed for fmk:" << flag.fmkIn;
      return nullptr;
    }
    func_graph = model_parser_->Parse(flag.modelFile, flag.weightFile);
  }
  if (UpdateFuncGraphInputsAndOutputsDtype(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "update graph inputs and outputs dtype failed.";
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
  std::unique_ptr<converter::Flags> flags(new (std::nothrow) converter::Flags);
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
