/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_JSON_DATA_HELPER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_JSON_DATA_HELPER_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "./securec.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

/// \brief Simple class to do data manipulation, contains helper function to update json files in dataset
class JsonHelper {
 public:
  /// \brief constructor
  JsonHelper() {}

  /// \brief Destructor
  ~JsonHelper() = default;

  /// \brief Create an Album dataset while taking in a path to a image folder
  ///     Creates the output directory if doesn't exist
  /// \param[in] in_dir Image folder directory that takes in images
  /// \param[in] out_dir Directory containing output json files
  Status CreateAlbum(const std::string &in_dir, const std::string &out_dir);

  /// \brief Update a json file field with a vector of integers
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional input for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<std::string> &value,
                     const std::string &out_file = "");

  /// \brief Update a json file field with a vector of type T values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  template <typename T>
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<T> &value,
                     const std::string &out_file = "") {
    try {
      Path in = Path(in_file);
      nlohmann::json js;
      if (in.Exists()) {
        std::ifstream in(in_file);
        MS_LOG(INFO) << "Filename: " << in_file << ".";
        in >> js;
        in.close();
      }
      js[key] = value;
      MS_LOG(INFO) << "Write outfile is: " << js << ".";
      if (out_file == "") {
        std::ofstream o(in_file, std::ofstream::trunc);
        o << js;
        o.close();
      } else {
        std::ofstream o(out_file, std::ofstream::trunc);
        o << js;
        o.close();
      }
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Update json failed ");
    }
    return Status::OK();
  }

  /// \brief Update a json file field with a single value of of type T
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  template <typename T>
  Status UpdateValue(const std::string &in_file, const std::string &key, const T &value,
                     const std::string &out_file = "") {
    try {
      Path in = Path(in_file);
      nlohmann::json js;
      if (in.Exists()) {
        std::ifstream in(in_file);
        MS_LOG(INFO) << "Filename: " << in_file << ".";
        in >> js;
        in.close();
      }
      js[key] = value;
      MS_LOG(INFO) << "Write outfile is: " << js << ".";
      if (out_file == "") {
        std::ofstream o(in_file, std::ofstream::trunc);
        o << js;
        o.close();
      } else {
        std::ofstream o(out_file, std::ofstream::trunc);
        o << js;
        o.close();
      }
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Update json failed ");
    }
    return Status::OK();
  }

  /// \brief Template function to write tensor to file
  /// \param[in] in_file File to write to
  /// \param[in] data Array of type T values
  /// \return Status The status code returned
  template <typename T>
  Status WriteBinFile(const std::string &in_file, const std::vector<T> &data) {
    try {
      std::ofstream o(in_file, std::ios::binary | std::ios::out);
      if (!o.is_open()) {
        RETURN_STATUS_UNEXPECTED("Error opening Bin file to write");
      }
      size_t length = data.size();
      o.write(reinterpret_cast<const char *>(&data[0]), std::streamsize(length * sizeof(T)));
      o.close();
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Write bin file failed ");
    }
    return Status::OK();
  }

  /// \brief Write pointer to bin, use pointer to avoid memcpy
  /// \param[in] in_file File name to write to
  /// \param[in] data Pointer to data
  /// \param[in] length Length of values to write from pointer
  /// \return Status The status code returned
  template <typename T>
  Status WriteBinFile(const std::string &in_file, T *data, size_t length) {
    try {
      std::ofstream o(in_file, std::ios::binary | std::ios::out);
      if (!o.is_open()) {
        RETURN_STATUS_UNEXPECTED("Error opening Bin file to write");
      }
      o.write(reinterpret_cast<const char *>(data), std::streamsize(length * sizeof(T)));
      o.close();
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Write bin file failed ");
    }
    return Status::OK();
  }

  /// \brief Helper function to copy content of a tensor to buffer
  /// \note This function iterates over the tensor in bytes, since
  /// \param[in] tensor_addr The memory held by a tensor, e.g. tensor->GetBuffer()
  /// \param[in] tensor_size The amount of data in bytes in tensor_addr, e.g. tensor->SizeInBytes()
  /// \param[out] addr The address to copy tensor data to
  /// \param[in] buffer_size The buffer size of addr
  /// \return The size of the tensor (bytes copied
  size_t DumpData(const unsigned char *tensor_addr, const size_t &tensor_size, void *addr, const size_t &buffer_size);

  /// \brief Helper function to delete key in json file
  /// note This function will return okay even if key not found
  /// \param[in] in_file Json file to remove key from
  /// \param[in] key The key to remove
  /// \return Status The status code returned
  Status RemoveKey(const std::string &in_file, const std::string &key, const std::string &out_file = "");

  /// \brief A print method typically used for debugging
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const;

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out Reference to the output stream being overloaded
  /// \param ds Reference to the DataSchema to display
  /// \return The output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const JsonHelper &dh) {
    dh.Print(out);
    return out;
  }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_JSON_HELPER_H_
