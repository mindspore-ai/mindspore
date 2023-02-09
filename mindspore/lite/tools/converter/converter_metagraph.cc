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

#include "tools/converter/converter_metagraph.h"
#include <vector>
#include <string>
#include "ops/op_utils.h"
#include "src/common/log_adapter.h"
#include "tools/common/func_graph_utils.h"
#include "tools/common/meta_graph_serializer.h"
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#include "tools/common/string_util.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"
#include "tools/optimizer/format/to_nhwc_format.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kFlatbuffersBuilderInitSize = 1024;
constexpr size_t kEncMaxLen = 16;
}  // namespace

STATUS ConverterToMetaGraph::UpdateMetaGraphOutputName(schema::MetaGraphT *meta_graph,
                                                       const std::vector<std::string> &output_names) {
  MS_CHECK_TRUE_MSG(meta_graph != nullptr, RET_NULL_PTR, "meta_graph is nullptr");
  if (output_names.size() != meta_graph->outputIndex.size()) {
    MS_LOG(ERROR) << "the num of setting output_names is greater than actual, " << output_names.size() << " > "
                  << meta_graph->outputIndex.size() << ".";
    return RET_ERROR;
  }
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    auto &tensor = meta_graph->allTensors.at(meta_graph->outputIndex.at(idx));
    tensor->name = output_names.at(idx);
  }
  return RET_OK;
}

STATUS ConverterToMetaGraph::UnifyFuncGraphFormat(const std::shared_ptr<ConverterPara> &param,
                                                  const FuncGraphPtr &old_graph) {
  auto value = old_graph->get_attr(ops::kFormat);
  if (value != nullptr && GetValue<int64_t>(value) != mindspore::NHWC) {
    auto format_pass = std::make_shared<opt::ToNHWCFormat>(param->fmk_type, param->train_model);
    MS_CHECK_TRUE_RET(format_pass != nullptr, RET_ERROR);
    if (!format_pass->Run(old_graph)) {
      MS_LOG(ERROR) << "Run ToNHWCFormat pass failed";
      return RET_ERROR;
    }
    auto transpose_pass = std::make_shared<opt::DecreaseTransposeAlgo>(param->fmk_type, param->train_model);
    MS_CHECK_TRUE_RET(transpose_pass != nullptr, RET_ERROR);
    if (!transpose_pass->Run(old_graph)) {
      MS_LOG(ERROR) << "Run DecreaseTransposeAlgo pass failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

schema::MetaGraphT *ConverterToMetaGraph::Build(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
  graphkernel::GraphKernelOptimize(func_graph, param);
#endif
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "make func graph failed.";
    return nullptr;
  }
  auto output_tensor_name = FuncGraphUtils::GetFuncGraphOutputNames(func_graph);
  if (output_tensor_name.empty()) {
    MS_LOG(ERROR) << "GetFuncGraphOutputName failed, Can not find output name.";
    return nullptr;
  }

  auto status = UnifyFuncGraphFormat(param, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "unify func graph format failed " << status;
    return nullptr;
  }

  // protobuf -> flatbuffer
  auto meta_graph = Export(func_graph, false, false, param->train_model);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return nullptr;
  }

  GraphDefTransform metagraph_transform;
  metagraph_transform.SetGraphDef(meta_graph);
  status = metagraph_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform meta graph failed " << status;
    delete meta_graph;
    return nullptr;
  }

  // output name will be modified by Transform
  status = UpdateMetaGraphOutputName(meta_graph, output_tensor_name);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    delete meta_graph;
    return nullptr;
  }

  return meta_graph;
}

STATUS ConverterToMetaGraph::Save(schema::MetaGraphT *meta_graph, const std::shared_ptr<ConverterPara> &param,
                                  void **model_data, size_t *data_size, bool not_save) {
  if (not_save) {
    flatbuffers::FlatBufferBuilder builder(kFlatbuffersBuilderInitSize);
    auto packed_buffer = MetaGraphSerializer::GetMetaGraphPackedBuff(&builder, *meta_graph, data_size);
    auto buffer = malloc(*data_size);
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "malloc failed.";
      return RET_ERROR;
    }

    if (memcpy_s(buffer, *data_size, packed_buffer, *data_size) != EOK) {
      free(buffer);
      MS_LOG(ERROR) << "memory copy failed.";
      return RET_ERROR;
    }
    *model_data = buffer;
  } else {
    unsigned char encKey[kEncMaxLen] = {0};
    size_t keyLen = 0;
    auto status = InitEncryptKey(param, encKey, &keyLen);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "check encryption failed.";
      return status;
    }

    status = MetaGraphSerializer::Save(*meta_graph, param->output_file, encKey, keyLen, param->encrypt_mode);
    (void)memset_s(encKey, kEncMaxLen, 0, kEncMaxLen);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "SAVE GRAPH FAILED:" << status << " " << GetErrorInfo(status);
      return status;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
