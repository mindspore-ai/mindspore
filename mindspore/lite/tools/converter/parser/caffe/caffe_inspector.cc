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

#include "tools/converter/parser/caffe/caffe_inspector.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
STATUS CaffeInspector::InspectModel(const caffe::NetParameter &proto) {
  net = proto;

  if (proto.layer_size() == 0) {
    MS_LOG(ERROR) << "net layer num is zero, prototxt file may be invalid.";
    return RET_ERROR;
  }

  ParseInput();

  SetLayerTopsAndBottoms();

  FindGraphInputsAndOutputs();

  return RET_OK;
}

STATUS CaffeInspector::ParseInput() {
  if (net.input_size() > 0) {
    MS_LOG(INFO) << "This net exist input.";
    for (int i = 0; i < net.input_size(); i++) {
      graphInput.insert(net.input(i));
    }
  }
  return RET_OK;
}

STATUS CaffeInspector::FindGraphInputsAndOutputs() {
  for (const auto &iter : layerBottoms) {
    if (!IsContain(layerTops, iter)) {
      graphInput.insert(iter);
    }
  }
  for (const auto &iter : layerTops) {
    if (layerBottoms.find(iter) == layerBottoms.end() && !IsContain(graphOutput, iter)) {
      graphOutput.push_back(iter);
    }
  }
  return RET_OK;
}

STATUS CaffeInspector::SetLayerTopsAndBottoms() {
  for (int32_t i = 0; i < net.layer_size(); i++) {
    auto &layer = const_cast<caffe::LayerParameter &>(net.layer(i));
    if (layer.top_size() == 1 && layer.bottom_size() == 1 && layer.top(0) == layer.bottom(0)) {
      continue;
    }
    if (layer.top_size() == 1 && layer.bottom_size() == 0) {
      graphInput.insert(layer.top(0));
    }
    for (int j = 0; j < layer.top_size(); j++) {
      if (!IsContain(layerTops, layer.top(j))) {
        layerTops.push_back(layer.top(j));
      }
    }
    for (int j = 0; j < layer.bottom_size(); j++) {
      layerBottoms.insert(layer.bottom(j));
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
