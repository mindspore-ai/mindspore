/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_CUSTOM_PASS_FILE_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_CUSTOM_PASS_FILE_UTILS_H_

#include <string>
#include <vector>

namespace mindspore {
namespace opt {

/**
 * @brief Cross-platform file system utilities for custom pass plugin management.
 *
 * This module provides a set of platform-independent file system operations
 * that work consistently across Windows, Linux, and macOS platforms.
 */
class FileUtils {
 public:
  /**
   * @brief Check if a file or directory exists.
   * @param path The path to check.
   * @return true if the path exists, false otherwise.
   */
  static bool Exists(const std::string &path);

  /**
   * @brief Check if a path points to a directory.
   * @param path The path to check.
   * @return true if the path is a directory, false otherwise.
   */
  static bool IsDirectory(const std::string &path);

  /**
   * @brief Get the file extension from a path.
   * @param path The file path.
   * @return The file extension including the dot (e.g., ".so"), or empty string if no extension.
   */
  static std::string GetExtension(const std::string &path);

  /**
   * @brief Get the filename from a full path.
   * @param path The full file path.
   * @return The filename without directory path.
   */
  static std::string GetBasename(const std::string &path);

  /**
   * @brief List all files and subdirectories in a directory.
   * @param directory_path The directory to list.
   * @return Vector of full paths to files and subdirectories.
   */
  static std::vector<std::string> ListDirectory(const std::string &directory_path);

  /**
   * @brief Check if a file has a supported plugin extension.
   * @param path The file path to check.
   * @return true if the file has .so, .dll, or .dylib extension.
   */
  static bool IsSupportedPluginFile(const std::string &path);
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_CUSTOM_PASS_FILE_UTILS_H_
