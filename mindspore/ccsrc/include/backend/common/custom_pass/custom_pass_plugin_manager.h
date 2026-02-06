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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "include/backend/common/custom_pass/custom_pass_plugin.h"
#include "include/backend/common/pass_manager/pass_manager.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {

class BACKEND_COMMON_EXPORT CustomPassPluginManager {
 public:
  static CustomPassPluginManager &GetInstance() {
    static CustomPassPluginManager instance;
    return instance;
  }

  bool LoadPlugin(const std::string &plugin_path, const std::string &pass_name, const std::string &device = "all",
                  const std::string &stage = "");

  void UnloadPlugin(const std::string &plugin_name);

  void UnloadAllPlugins();

  std::vector<std::shared_ptr<CustomPassPlugin>> GetAllPlugins();

  std::shared_ptr<CustomPassPlugin> GetPlugin(const std::string &plugin_name);

  // Device filtering enables selective pass registration for different hardware targets
  void RegisterPassesToOptimizer(std::shared_ptr<GraphOptimizer> optimizer, const std::string &device = "all");

  bool IsPluginLoaded(const std::string &plugin_name) const;

  std::vector<std::string> GetLoadedPluginNames() const;

  // Public methods for PluginInfo cleanup
  void DestroyPluginInstance(void *handle, CustomPassPlugin *plugin);
  void CloseDynamicLibrary(void *handle);

 private:
  CustomPassPluginManager() : next_registration_order_(0) {}
  ~CustomPassPluginManager() = default;

  // Singleton pattern requires disabled copy operations
  CustomPassPluginManager(const CustomPassPluginManager &) = delete;
  CustomPassPluginManager &operator=(const CustomPassPluginManager &) = delete;

  using CreatePluginFunc = CustomPassPlugin *(*)();
  using DestroyPluginFunc = void (*)(CustomPassPlugin *);

  // Avoid code duplication across multiple unload scenarios
  bool UnloadPluginInternal(const std::string &plugin_name);

  // Abstract platform-specific dynamic library operations for portability
  void *LoadDynamicLibrary(const std::string &plugin_path);
  CreatePluginFunc GetCreatePluginFunction(void *handle, const std::string &plugin_path);

  // Helper methods for pass validation and registration
  bool ValidatePassExists(const std::string &pass_name, const std::string &plugin_name,
                          const std::vector<std::string> &available_passes);
  std::string FormatAvailablePassesList(const std::vector<std::string> &available_passes);
  void RegisterPassExecution(const std::string &pass_name, const std::string &plugin_name, const std::string &device,
                             const std::string &stage);

  // RAII wrapper for dynamic library handle
  struct LibraryHandle {
    void *handle;
    explicit LibraryHandle(void *h) : handle(h) {}
    ~LibraryHandle() {
      if (handle) {
        auto &manager = CustomPassPluginManager::GetInstance();
        manager.CloseDynamicLibrary(handle);
      }
    }
    LibraryHandle(const LibraryHandle &) = delete;
    LibraryHandle &operator=(const LibraryHandle &) = delete;
  };

  // Unified plugin information structure with safe reference management
  struct PluginInfo {
    std::string plugin_name;                       // Plugin name
    std::string plugin_path;                       // Plugin file path
    std::shared_ptr<CustomPassPlugin> plugin;      // Plugin instance with proper lifetime management
    void *handle;                                  // Dynamic library handle (for manager use)
    std::shared_ptr<LibraryHandle> shared_handle;  // Shared library handle for safe cleanup
    std::vector<std::string> available_passes;     // Passes provided by this plugin

    PluginInfo(const std::string &name, const std::string &path, std::shared_ptr<CustomPassPlugin> p, void *h,
               std::shared_ptr<LibraryHandle> sh)
        : plugin_name(name), plugin_path(path), plugin(p), handle(h), shared_handle(sh) {
      if (plugin) {
        try {
          available_passes = plugin->GetAvailablePassNames();
        } catch (const std::exception &e) {
          // Handle exception gracefully
          available_passes.clear();
        }
      }
    }

    // Disable copy, only allow move
    PluginInfo(const PluginInfo &) = delete;
    PluginInfo &operator=(const PluginInfo &) = delete;
    PluginInfo(PluginInfo &&other) noexcept
        : plugin_name(std::move(other.plugin_name)),
          plugin_path(std::move(other.plugin_path)),
          plugin(std::move(other.plugin)),
          handle(other.handle),
          shared_handle(std::move(other.shared_handle)),
          available_passes(std::move(other.available_passes)) {
      other.handle = nullptr;  // Transfer ownership
    }
    PluginInfo &operator=(PluginInfo &&other) noexcept {
      if (this != &other) {
        CleanupResources();
        plugin_name = std::move(other.plugin_name);
        plugin_path = std::move(other.plugin_path);
        plugin = std::move(other.plugin);
        handle = other.handle;
        shared_handle = std::move(other.shared_handle);
        available_passes = std::move(other.available_passes);
        other.handle = nullptr;  // Transfer ownership
      }
      return *this;
    }

    ~PluginInfo() { CleanupResources(); }

   private:
    void CleanupResources();
  };

  // Enhanced pass registration information
  struct PassExecution {
    std::string pass_name;      // Pass name
    std::string plugin_name;    // Plugin name (key improvement)
    std::string device;         // Device type
    std::string stage;          // Execution stage
    size_t registration_order;  // Registration order for A-B-A support

    PassExecution(const std::string &pass, const std::string &plugin, const std::string &dev, const std::string &st,
                  size_t order)
        : pass_name(pass), plugin_name(plugin), device(dev), stage(st), registration_order(order) {}
  };

  // Main data structures
  std::map<std::string, std::unique_ptr<PluginInfo>> plugins_;  // Plugin name -> Plugin info
  std::vector<PassExecution> pass_execution_order_;             // Pass execution order
  size_t next_registration_order_;                              // Global registration counter
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_MANAGER_H_
