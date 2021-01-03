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
#include <utility>
#include "tools/converter/converter_flags.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"
#include "ir/func_graph.h"

#include "src/common/log_adapter.h"
#include "tools/common/storage.h"
#include "parser/caffe/caffe_converter.h"
#include "parser/tflite/tflite_converter.h"
#include "parser/onnx/onnx_converter.h"
#include "parser/tf/tf_converter.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/anf_importer/import_from_mindir.h"
#include "proto/onnx.pb.h"
#include "include/version.h"

namespace mindspore {
namespace lite {
using FmkType = converter::FmkType;
static const char *DELIM_SLASH = "/";
Converter::Converter() {
  this->transform = new GraphDefTransform;
  this->anfTransform = new AnfTransform;
}

Converter::~Converter() {
  delete modelParser;
  delete modelImporter;
  delete transform;
  delete anfTransform;
}

class MindsporeImporter : public Converter {
 public:
  MindsporeImporter() { modelImporter = new AnfImporterFromMindir(); }

  ~MindsporeImporter() override = default;
};

MetaGraphT *Converter::Convert(const converter::Flags *flag) {
  // parse the model and weight file to generate inference data structure
  FuncGraphPtr graph = nullptr;
  if (flag->fmk == converter::FmkType_MS) {
    MS_ASSERT(nullptr != modelImporter);
    int status = modelImporter->Import(flag);
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    graph = modelImporter->GetResult();
    if (graph == nullptr) {
      return nullptr;
    }
    graph->set_attr("graph_name", MakeValue("main_graph"));
    graph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_MS)));
  } else {
    MS_ASSERT(nullptr != modelParser);
    const std::string modelFile = flag->modelFile;
    const std::string weightFile = flag->weightFile;
    graph = modelParser->Parse(modelFile, weightFile, flag->quantType);
  }
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Parser/Import model return nullptr";
    return nullptr;
  }

  graph = anfTransform->Transform(graph, flag);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Transform anf graph return nullptr";
    return nullptr;
  }

  // anf -- fb
  auto meta_graph = Export(graph);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return nullptr;
  }

  // transform
  transform->SetGraphDef(meta_graph);
  auto status = transform->Transform(*flag);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform meta graph failed " << status;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  return meta_graph;
}

int RunConverter(int argc, const char **argv) {
  std::unique_ptr<converter::Flags> flags(new (std::nothrow) converter::Flags);
  if (flags == nullptr) {
    MS_LOG(ERROR) << "NEW FLAGS ERROR:" << RET_MEMORY_FAILED << " " << GetErrorInfo(RET_MEMORY_FAILED);
    std::cout << "NEW FLAGS ERROR:" << RET_MEMORY_FAILED << " " << GetErrorInfo(RET_MEMORY_FAILED) << std::endl;
    return RET_MEMORY_FAILED;
  }
  auto status = flags->Init(argc, argv);
  if (status != RET_OK) {
    if (status != RET_SUCCESS_EXIT) {
      MS_LOG(ERROR) << "CONVERTER::FLAGS INIT FAILED:" << status << " " << GetErrorInfo(status) << std::endl;
      std::cout << "CONVERTER::FLAGS INIT FAILED:" << status << " " << GetErrorInfo(status) << std::endl;
    }
    std::cout << GetErrorInfo(status) << std::endl;
    return status;
  }
  // Load graph
  std::string modelName = flags->modelFile.substr(flags->modelFile.find_last_of(DELIM_SLASH) + 1);
  MS_LOG(INFO) << "start reading model file";

  MetaGraphT *fb_graph = nullptr;
  switch (flags->fmk) {
    case FmkType::FmkType_MS: {
      MindsporeImporter mindsporeImporter;
      fb_graph = mindsporeImporter.Convert(flags.get());
      break;
    }
    case FmkType::FmkType_CAFFE: {
      CaffeConverter caffeConverter;
      fb_graph = caffeConverter.Convert(flags.get());
    } break;
    case FmkType::FmkType_TFLITE: {
      TfliteConverter tfLiteConverter;
      fb_graph = tfLiteConverter.Convert(flags.get());
    } break;
    case FmkType::FmkType_ONNX: {
      OnnxConverter onnxConverter;
      fb_graph = onnxConverter.Convert(flags.get());
    } break;
    case FmkType::FmkType_TF: {
      TFConverter tfConverter;
      fb_graph = tfConverter.Convert(flags.get());
    } break;
    default: {
      MS_LOG(ERROR) << "UNSUPPORTED FMKTYPE " << flags->fmk << ":" << RET_INPUT_PARAM_INVALID << " "
                    << GetErrorInfo(RET_INPUT_PARAM_INVALID);
      std::cout << "UNSUPPORTED FMKTYPE " << flags->fmk << ":" << RET_INPUT_PARAM_INVALID << " "
                << GetErrorInfo(RET_INPUT_PARAM_INVALID) << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  NoSupportOp::GetInstance()->PrintOps();
  status = ReturnCode::GetSingleReturnCode()->GetReturnCode();
  if (fb_graph == nullptr) {
    MS_LOG(ERROR) << "CONVERT RESULT FAILED:" << status << " " << GetErrorInfo(status);
    std::cout << "CONVERT RESULT FAILED:" << status << " " << GetErrorInfo(status) << std::endl;
    return status;
  }

  //   save graph to file
  Storage storage;
  fb_graph->version = Version();
  status = storage.Save(*fb_graph, flags->outputFile);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SAVE GRAPH FAILED:" << status << " " << GetErrorInfo(status);
    std::cout << "SAVE GRAPH FAILED:" << status << " " << GetErrorInfo(status) << std::endl;
    return status;
  }

  delete fb_graph;
  MS_LOG(INFO) << "CONVERT RESULT SUCCESS:" << status;
  std::cout << "CONVERT RESULT SUCCESS:" << status << std::endl;
  return status;
}
}  // namespace lite
}  // namespace mindspore
