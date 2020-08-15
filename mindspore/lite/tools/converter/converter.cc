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
#include <vector>
#include <utility>
#include <memory>
#include "tools/converter/converter_flags.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"
#include "ir/func_graph.h"

#include "utils/log_adapter.h"
#include "tools/common/storage.h"
#include "parser/caffe/caffe_converter.h"
#include "parser/tflite/tflite_converter.h"
#include "parser/onnx/onnx_converter.h"
#include "src/common/anf_exporter/anf_exporter.h"
#include "src/common/anf_importer/import_from_protobuf.h"
#include "tools/converter/parser/onnx/onnx.pb.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"

namespace mindspore {
namespace lite {
using FmkType = converter::FmkType;
Converter::Converter() {
  this->transform = new GraphDefTransform;
  this->anfTransform = new AnfTransform;
}

Converter::~Converter() {
  if (nullptr != modelParser) {
    delete modelParser;
  }
  if (nullptr != modelImporter) {
    delete modelImporter;
  }
  if (nullptr != transform) {
    delete transform;
  }
  if (nullptr != anfTransform) {
    delete anfTransform;
  }
}

class MindsporeImporter : public Converter {
 public:
  MindsporeImporter(onnx::ModelProto *onnx_model, FuncGraphPtr func_graph) {
    modelImporter = new AnfImporterFromProtobuf(onnx_model, std::move(func_graph));
  }

  ~MindsporeImporter() override = default;
};

MetaGraphT *Converter::Convert(const converter::Flags *flag) {
  // parse the model and weight file to generate inference data structure
  FuncGraphPtr graph = nullptr;
  if (flag->fmk == converter::FmkType_MS) {
    MS_ASSERT(nullptr != modelImporter);
    modelImporter->Import(flag->quantType);
    graph = modelImporter->GetResult();
  } else {
    MS_ASSERT(nullptr != modelParser);
    const std::string modelFile = flag->modelFile;
    const std::string weightFile = flag->weightFile;
    auto meta_graph = modelParser->Parse(modelFile, weightFile, flag->quantType);
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "Parse to metaGraph return nullptr";
      return nullptr;
    }
    // todo hangangqiang
    graph = ModelParser::Fb2Anf(meta_graph);
  }
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Parser/Import model return nullptr";
    return nullptr;
  }

  graph = anfTransform->Transform(graph);

  CreateQuantizer(graph, flag);
  if (mQuantizer != nullptr) {
    mQuantizer->flags = *flag;
    auto status = mQuantizer->DoQuantize(graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant failed " << status;
      return nullptr;
    }
    quant::QuantCast quant_cast;
    quant_cast.SetInputDataDType(kNumberTypeFloat32);
    status = quant_cast.Run(graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "add QuantCast error";
      return nullptr;
    }
  }

  // anf -- fb
  auto meta_graph = Export(graph);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph return nullptr";
    return nullptr;
  }

  // transform
  transform->SetGraphDef(meta_graph);
  transform->CreateQuantizer(flag);
  auto status = transform->Transform(*flag);
  if (status != 0) {
    MS_LOG(ERROR) << "FBTransform model failed " << status;
    return nullptr;
  }
  return meta_graph;
}

void Converter::CreateQuantizer(FuncGraphPtr funcGraph, const converter::Flags *flags) {
  auto type = flags->quantType;
  switch (type) {
    case mindspore::schema::QuantType_AwareTraining: {
      // mQuantizer.reset(new AwareQuantizer(graphDefT, flags->inputInferenceTypeIn, flags->stdDev, flags->mean));
      break;
    }
      //    case mindspore::schema::QuantType_WeightQuant: {
      //      MS_LOG(INFO) << "create WeightQuantizer!";
      //      mQuantizer.reset(
      //        new quant::WeightQuantizer(funcGraph, flags->quantSize, flags->convWeightQuantChannelThreshold,
      //        flags->bitNum));
      //      break;
      //    }
    case mindspore::schema::QuantType_PostTraining: {
      MS_LOG(INFO) << "create PostTrainningQuantizer!";
      mQuantizer.reset(new quant::PostTrainingQuantizer(funcGraph, flags->configFile, 8));
      break;
    }
    case mindspore::schema::QuantType_QUANT_NONE:
      MS_LOG(INFO) << "Not do quantization for model!";
      break;
    default:
      MS_LOG(INFO) << "will support quntizer type " << flags->quantTypeIn.c_str() << " in the future!";
      break;
  }
}
int RunConverter(int argc, const char **argv) {
  auto flags = new converter::Flags;
  auto status = flags->Init(argc, argv);
  if (status == RET_SUCCESS_EXIT) {
    return 0;
  }
  if (status != 0) {
    MS_LOG(ERROR) << "converter::Flags Init failed: " << status;
    return 1;
  }
  // Load graph
  std::string modelName = flags->modelFile.substr(flags->modelFile.find_last_of(DELIM_SLASH) + 1);
  MS_LOG(INFO) << "start reading model file";

  MetaGraphT *fb_graph = nullptr;
  switch (flags->fmk) {
    case FmkType::FmkType_MS: {
      auto graph = std::make_shared<FuncGraph>();
      auto onnx_graph = AnfImporterFromProtobuf::ReadOnnxFromBinary(flags->modelFile);
      MindsporeImporter mindsporeImporter(onnx_graph, graph);
      fb_graph = mindsporeImporter.Convert(flags);
      break;
    }
    case FmkType::FmkType_CAFFE: {
      CaffeConverter caffeConverter;
      fb_graph = caffeConverter.Convert(flags);
    } break;
    case FmkType::FmkType_TFLITE: {
      TfliteConverter tfLiteConverter;
      fb_graph = tfLiteConverter.Convert(flags);
    } break;
    case FmkType::FmkType_ONNX: {
      OnnxConverter onnxConverter;
      fb_graph = onnxConverter.Convert(flags);
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported fmkType: " << flags->fmk;
      return 1;
    }
  }
  if (fb_graph == nullptr) {
    MS_LOG(ERROR) << "Convert model return nullptr";
    return 1;
  }

  //   save graph to file
  Storage storage;
  status = storage.Save(*fb_graph, flags->outputFile);
  if (status != 0) {
    MS_LOG(ERROR) << "Save graph failed";
    return 1;
  }
  MS_LOG(INFO) << "CONVERT RESULT: SUCCESS!";

  return 0;
}
}  // namespace lite
}  // namespace mindspore
