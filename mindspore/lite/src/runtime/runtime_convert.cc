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

#ifdef RUNTIME_CONVERT
#include "src/runtime/runtime_convert.h"
#include "tools/common/graph_util.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/anf_transform.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/converter/import/mindspore_importer.h"

namespace mindspore::lite {
char *RuntimeConvert(const char *file_path, size_t *size) {
  void *model_buf = nullptr;
  converter::Flags flag;
  flag.fmk = converter::kFmkTypeMs;
  flag.modelFile = file_path;
  flag.inputDataType = kTypeUnknown;
  flag.outputDataType = kTypeUnknown;
  flag.saveFP16 = false;
  flag.trainModel = false;

  MindsporeImporter ms_import;
  FuncGraphPtr func_graph = ms_import.ImportMindIR(flag);
  if (func_graph == nullptr) {
    return nullptr;
  }

  if (UpdateFuncGraphInputsAndOutputsDtype(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "Update graph inputs and outputs dtype failed.";
    return nullptr;
  }

  // funcgraph compile
  AnfTransform funcgraph_transform;
  func_graph = funcgraph_transform.Transform(func_graph, &flag);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Transform anf graph return nullptr";
    return nullptr;
  }

  // protobuf -> flatbuffer
  auto meta_graph = Export(func_graph, false, false, false);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return nullptr;
  }

  // metagraph compile
  GraphDefTransform metagraph_transform;
  metagraph_transform.SetGraphDef(meta_graph);
  auto status = metagraph_transform.Transform(flag);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform meta graph failed " << status;
    delete meta_graph;
    return nullptr;
  }

  status = UpdateGraphOutputName(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    delete meta_graph;
    return nullptr;
  }

  meta_graph->version = Version();
  status = TransferMetaGraph(*meta_graph, &model_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(model_buf);
}
}  // namespace mindspore::lite
#endif
