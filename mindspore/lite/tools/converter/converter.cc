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
#include "tools/converter/converter_flags.h"
#include "src/common/log_adapter.h"
#include "tools/common/storage.h"
#include "parser/caffe/caffe_converter.h"
#include "parser/tflite/tflite_converter.h"
#include "parser/onnx/onnx_converter.h"
#include "parser/tf/tf_converter.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "include/version.h"
#include "src/train/train_populate_parameter.h"

namespace mindspore {
namespace lite {
using FmkType = converter::FmkType;

MindsporeImporter::MindsporeImporter() { kernel::PopulateTrainParameters(); }

std::unique_ptr<Converter> Converter::CreateConverter(converter::FmkType fmk) {
  switch (fmk) {
    case FmkType::FmkType_MS:
      return std::make_unique<MindsporeImporter>();
    case FmkType::FmkType_CAFFE:
      return std::make_unique<CaffeConverter>();
    case FmkType::FmkType_TFLITE:
      return std::make_unique<TfliteConverter>();
    case FmkType::FmkType_ONNX:
      return std::make_unique<OnnxConverter>();
    case FmkType::FmkType_TF:
      return std::make_unique<TFConverter>();
    default: {
      return nullptr;
    }
  }
}

MetaGraphT *Converter::Convert(const std::unique_ptr<converter::Flags> &flag) {
  if (flag == nullptr) {
    MS_LOG(ERROR) << "Input flag is nullptr";
    return nullptr;
  }
  auto graph = BuildFuncGraph(flag->modelFile, flag->weightFile, flag->quantType);
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
  MS_LOG(INFO) << "Run anfTransform success";

  // protobuf -> flatbuf
  auto meta_graph = Export(graph, false, false, flag->trainModel);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return nullptr;
  }
  MS_LOG(INFO) << "export success";

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
  auto converter = Converter::CreateConverter(flags->fmk);
  if (converter == nullptr) {
    oss.clear();
    oss << "UNSUPPORTED FMKTYPE " << flags->fmk << ":" << RET_INPUT_PARAM_INVALID << " "
        << GetErrorInfo(RET_INPUT_PARAM_INVALID);
    MS_LOG(ERROR) << oss.str();
    std::cout << oss.str() << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  auto meta_graph = converter->Convert(flags);
  NoSupportOp::GetInstance()->PrintOps();
  status = ReturnCode::GetSingleReturnCode()->GetReturnCode();
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
