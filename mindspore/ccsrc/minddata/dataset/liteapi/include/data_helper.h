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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATA_HELPER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATA_HELPER_H_

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"

namespace mindspore {
namespace dataset {

/// \brief Simple class to do data manipulation, contains helper function to update json files in dataset
class DataHelper {
 public:
  /// \brief constructor
  DataHelper() {}

  /// \brief Destructor
  ~DataHelper() = default;

  /// \brief Create an Album dataset while taking in a path to a image folder
  ///     Creates the output directory if doesn't exist
  /// \param[in] in_dir Image folder directory that takes in images
  /// \param[in] out_dir Directory containing output json files
  Status CreateAlbum(const std::string &in_dir, const std::string &out_dir) {
    return CreateAlbumIF(StringToChar(in_dir), StringToChar(out_dir));
  }

  /// \brief Update a json file field with a vector of string values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional input for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<std::string> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), VectorStringToChar(value), StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of bool values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<bool> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of int8 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<int8_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of uint8 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<uint8_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of int16 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<int16_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of uint16 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<uint16_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of int32 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<int32_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of uint32 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<uint32_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of int64 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<int64_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of uint64 values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<uint64_t> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of float values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<float> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a vector of double values
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value array to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateArray(const std::string &in_file, const std::string &key, const std::vector<double> &value,
                     const std::string &out_file = "") {
    return UpdateArrayIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a string value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const std::string &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), StringToChar(value), StringToChar(out_file));
  }

  /// \brief Update a json file field with a bool value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const bool &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an int8 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const int8_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an uint8 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const uint8_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an int16 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const int16_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an uint16 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const uint16_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an int32 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const int32_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an uint32 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const uint32_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an int64 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const int64_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with an uint64 value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const uint64_t &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a float value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const float &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
  }

  /// \brief Update a json file field with a double value
  /// \param in_file The input file name to read in
  /// \param key Key of field to write to
  /// \param value Value to write to file
  /// \param out_file Optional parameter for output file path, will write to input file if not specified
  /// \return Status The status code returned
  Status UpdateValue(const std::string &in_file, const std::string &key, const double &value,
                     const std::string &out_file = "") {
    return UpdateValueIF(StringToChar(in_file), StringToChar(key), value, StringToChar(out_file));
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
        return Status(kMDUnexpectedError, "Error opening Bin file to write");
      }
      size_t length = data.size();
      o.write(reinterpret_cast<const char *>(&data[0]), std::streamsize(length * sizeof(T)));
      o.close();
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      return Status(kMDUnexpectedError, "Write bin file failed ");
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
        return Status(kMDUnexpectedError, "Error opening Bin file to write");
      }
      o.write(reinterpret_cast<const char *>(data), std::streamsize(length * sizeof(T)));
      o.close();
    }
    // Catch any exception and convert to Status return code
    catch (const std::exception &err) {
      return Status(kMDUnexpectedError, "Write bin file failed ");
    }
    return Status::OK();
  }

  /// \brief Helper function to copy content of a tensor to buffer
  /// \note This function iterates over the tensor in bytes, since
  /// \param[in] tensor_addr The memory held by a tensor
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
  Status RemoveKey(const std::string &in_file, const std::string &key, const std::string &out_file = "") {
    return RemoveKeyIF(StringToChar(in_file), StringToChar(key), StringToChar(out_file));
  }

  /// \brief A print method typically used for debugging
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const;

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out Reference to the output stream being overloaded
  /// \param ds Reference to the DataSchema to display
  /// \return The output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DataHelper &dh) {
    dh.Print(out);
    return out;
  }

 private:
  // Helper function for dual ABI support
  Status CreateAlbumIF(const std::vector<char> &in_dir, const std::vector<char> &out_dir);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<std::vector<char>> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<bool> &value,
                       const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<int8_t> &value,
                       const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<uint8_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<int16_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<uint16_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<int32_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<uint32_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<int64_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                       const std::vector<uint64_t> &value, const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<float> &value,
                       const std::vector<char> &out_file);
  Status UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<double> &value,
                       const std::vector<char> &out_file);

  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<char> &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const bool &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int8_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint8_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int16_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint16_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int32_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint32_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int64_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint64_t &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const float &value,
                       const std::vector<char> &out_file);
  Status UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const double &value,
                       const std::vector<char> &out_file);
  Status RemoveKeyIF(const std::vector<char> &in_file, const std::vector<char> &key, const std::vector<char> &out_file);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATA_HELPER_H_
