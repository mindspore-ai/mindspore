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
#include "minddata/dataset/engine/serdes.h"

#include "debug/common.h"
#include "utils/utils.h"

namespace mindspore {
namespace dataset {

Status Serdes::SaveToJSON(std::shared_ptr<DatasetNode> node, const std::string &filename, nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(out_json);
  // Dump attributes of current node to json string
  nlohmann::json args;
  RETURN_IF_NOT_OK(node->to_json(&args));
  args["op_type"] = node->Name();

  // If the current node isn't leaf node, visit all its children and get all attributes
  std::vector<nlohmann::json> children_pipeline;
  if (!node->IsLeaf()) {
    for (auto child : node->Children()) {
      nlohmann::json child_args;
      RETURN_IF_NOT_OK(SaveToJSON(child, "", &child_args));
      children_pipeline.push_back(child_args);
    }
  }
  args["children"] = children_pipeline;

  // Save json string into file if filename is given.
  if (!filename.empty()) {
    RETURN_IF_NOT_OK(SaveJSONToFile(args, filename));
  }

  *out_json = args;
  return Status::OK();
}

Status Serdes::SaveJSONToFile(nlohmann::json json_string, const std::string &file_name) {
  try {
    auto realpath = Common::GetRealPath(file_name);
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed, path=" << file_name;
      RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + file_name);
    }

    std::ofstream file(realpath.value());
    file << json_string;
    file.close();

    ChangeFileMode(realpath.value(), S_IRUSR | S_IWUSR);
  } catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Save json string into " + file_name + " failed!");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
