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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \brief The Serdes class is used to serialize an IR tree into JSON string and dump into file if file name
/// specified.
class Serdes {
 public:
  /// \brief Constructor
  Serdes() {}

  /// \brief default destructor
  ~Serdes() = default;

  /// \brief function to serialize IR tree into JSON string and/or JSON file
  /// \param[in] node IR node to be transferred
  /// \param[in] filename The file name. If specified, save the generated JSON string into the file
  /// \param[out] out_json The result json string
  /// \return Status The status code returned
  Status SaveToJSON(std::shared_ptr<DatasetNode> node, const std::string &filename, nlohmann::json *out_json);

 protected:
  /// \brief Helper function to save JSON to a file
  /// \param[in] json_string The JSON string to be saved to the file
  /// \param[in] file_name The file name
  /// \return Status The status code returned
  Status SaveJSONToFile(nlohmann::json json_string, const std::string &file_name);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_
