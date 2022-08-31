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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_INCLUDE_SHARED_INCLUDE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_INCLUDE_SHARED_INCLUDE_H_

#include <map>
#include <set>
#include <string>
#include <vector>
/***
 * This file is is complied with both MindData and plugin separately. Changing this file without compiling both
 * projects could lead to undefined behaviors.
 */

namespace mindspore {
namespace dataset {
namespace plugin {
// forward declares
class PluginManagerBase;
class MindDataManagerBase;
// any plugin module is expected to expose these two functions as the entry point

/// \brief First handshake between plugin and MD.
/// \param[in] MindDataManagerBase, a pointer for callback functions. (plugin can call MD function)
/// \return status code, Status::OK() if function succeeds.
extern "C" PluginManagerBase *GetInstance(MindDataManagerBase *);

/// \brief Definition of this function is expected to deallocate PluginManager
/// \return void
extern "C" void DestroyInstance();

/***
 * Tentative version rule for Plugin: X.Y.Z
 * X, major version, increment when additional file is included or other major changes
 * Y, minor version, increment when class/API are changed or other minor changes
 * Z, patch version, increment when bug fix is introduced or other patches
 */
static constexpr char kSharedIncludeVersion[] = "0.5.6";

/***
 * All derived classes defined in plugin side needs to inherit from this.
 */
class PluginBase {
 protected:
  virtual ~PluginBase() noexcept = default;
};

/***
 * This class is used for callback. Functions defined in MindData can be exposed to plugin via this virtual class.
 * All derived classes of this have their definition on MindData side.
 */
class MindDataBase {
 protected:
  virtual ~MindDataBase() noexcept = default;
};

/***
 * This is a simplified version of Status code. It intends to offer a simple <bool,string> return type. The syntax of
 * this class is modelled after existing Status code.
 */
class Status : PluginBase {
 public:
  static Status OK() noexcept { return Status(); }
  static Status Error(const std::string &msg) noexcept { return Status(msg); }
  Status(const Status &) = default;
  Status(Status &&) = default;

  // helper functions
  bool IsOk() const noexcept { return success_; }
  const std::string &ToString() const noexcept { return status_msg_; }

 private:
  Status() noexcept : success_(true) {}
  explicit Status(const std::string &msg) noexcept : success_(false), status_msg_(msg) {}
  const bool success_;
  const std::string status_msg_;
};

/***
 * This is the interface through which MindData interacts with external .so files. There can only be 1 instance of
 * this class (hence the name Singleton) per so file. This class is the in-memory representation of each so file.
 * GetModule() returns class that contains runtime logic (e.g. GDALDecode). Raw pointer is used so that PluginManager
 * owns the lifetime of whatever objects it returns. MindData can not part-take in the memory management of plugin
 * objects. PluginManager is expected to be destroyed when DestroyInstance() is called.
 */
class PluginManagerBase : public PluginBase {
 public:
  virtual std::string GetPluginVersion() noexcept = 0;

  virtual std::map<std::string, std::set<std::string>> GetModuleNames() noexcept = 0;

  /// \brief return the module (e.g. a specific plugin tensor op) based on the module name. (names can be queried)
  /// \return pointer to the module. returns nullptr if module doesn't exist.
  virtual PluginBase *GetModule(const std::string &name) noexcept = 0;
};

/***
 * This class is used to get functions (e.g. Log) from MindData.
 */
class MindDataManagerBase : public MindDataBase {
 public:
  virtual MindDataBase *GetModule(const std::string &name) noexcept = 0;
};

/***
 * this struct is a Tensor in its simplest form, it is used to send Tensor data between MindData and Plugin.
 */
class Tensor : public PluginBase {
 public:
  std::vector<unsigned char> buffer_;  // contains the actual content of tensor
  std::vector<int64_t> shape_;         // shape of tensor, can be empty which means scalar
  std::vector<int64_t> offsets_;       // store the offsets for only string Tensor
  std::string type_;  // supported string literals "unknown", "bool", "int8", "uint8", "int16", "uint16", "int32",
                      // "uint32", "int64", "uint64", "float16", "float32", "float64", "string"
};

/***
 *  This is plugin's TensorOp which resembles MindData's TensorOp. No exception is allowed. Each function needs to catch
 *  all the errors thrown by 3rd party lib and if recovery isn't possible, return false and log the error. if MindData
 *  sees an function returns false, it will quit immediately without any attempt to resolve the issue.
 */
class TensorOp : public PluginBase {
 public:
  /// \brief Parse input params for this op. This function will only be called once for the lifetime of this object.
  /// \return status code, Status::OK() if function succeeds.
  virtual Status ParseSerializedArgs(const std::string &) noexcept = 0;

  /// \brief Perform operation on in_row and return out_row
  /// \param[in] in_row pointer to input tensor row
  /// \param[out] out_row pointer to output tensor row
  /// \return status code, Status::OK() if function succeeds.
  virtual Status Compute(std::vector<Tensor> *in_row, std::vector<Tensor> *out_row) noexcept = 0;
};

}  // namespace plugin
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_INCLUDE_SHARED_INCLUDE_H_
