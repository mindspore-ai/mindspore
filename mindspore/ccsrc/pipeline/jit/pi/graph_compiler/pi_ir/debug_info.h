/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_DEBUG_INFO__H_
#define MINDSPORE_PI_JIT_DEBUG_INFO__H_

#include <memory>
#include <string>

namespace mindspore {
namespace pijit {
namespace ir {
/// \brief DebugInfo is a container, which is used to store information such as line No. and file name etc.
class DebugInfo : public std::enable_shared_from_this<DebugInfo> {
 public:
  /**
   * \brief The constructor of DebugInfo.
   *
   * \param[in] desc The description of the host node.
   *
   * \return The instance of DebugInfo.
   */
  explicit DebugInfo(const std::string &desc) : DebugInfo(desc, "", 0) {}

  /**
   * \brief The constructor of DebugInfo.
   *
   * \param[in] desc The description of the host node.
   * \param[in] file_name The file name where the host is located.
   * \param[in] line_no The line number of the host node.
   *
   * \return The instance of DebugInfo.
   */
  DebugInfo(const std::string &desc, const std::string &file_name, int line_no)
      : desc_(desc), file_name_(file_name), line_no_(line_no) {}

  /// \brief Destructor.
  virtual ~DebugInfo() = default;

  /**
   * \brief Get the description of the host node.
   *
   * \return The description of the host node.
   */
  const std::string &GetDesc() const { return desc_; }

  /**
   * \brief Set the description of the host node..
   */
  void SetDesc(const std::string &desc) { desc_ = desc; }

  /**
   * \brief Get the file name where the host is located.
   *
   * \return The file name where the host is located.
   */
  const std::string &GetFileName() const { return file_name_; }

  /**
   * \brief Set the file name where the host is located.
   *
   * \param[in] file_name The file name where the host is located.
   */
  void SetFileName(const std::string &file_name) { file_name_ = file_name; }

  /**
   * \brief Get the line number of the host node.
   *
   * \return The line number of the host node.
   */
  int GetLineNo() const { return line_no_; }

  /**
   * \brief Set the line number of the host node.
   *
   * \param[in] line_no The line number of the host node.
   */
  void SetLineNo(int line_no) { line_no_ = line_no; }

 private:
  /// \brief The description of the host node.
  std::string desc_;
  /// \brief The file name where the host is located.
  std::string file_name_;
  /// \brief The line number of the host node.
  int line_no_;
};

using DebugInfoPtr = std::shared_ptr<DebugInfo>;
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_DEBUG_INFO__H_
