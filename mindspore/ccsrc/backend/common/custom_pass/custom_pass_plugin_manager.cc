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

#include "include/backend/common/custom_pass/custom_pass_plugin_manager.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "backend/common/custom_pass/file_utils.h"
#include "utils/log_adapter.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#else
#include <windows.h>
#endif

namespace mindspore {
namespace opt {

bool CustomPassPluginManager::LoadPlugin(const std::string &plugin_path, const std::string &pass_name,
                                         const std::string &device, const std::string &stage) {
  if (!FileUtils::Exists(plugin_path)) {
    MS_LOG(ERROR) << "Plugin file not found: " << plugin_path;
    return false;
  }

  std::string extension = FileUtils::GetExtension(plugin_path);
  if (!FileUtils::IsSupportedPluginFile(plugin_path)) {
    MS_LOG(ERROR) << "Unsupported plugin file extension: " << extension;
    return false;
  }

  // Load and create plugin instance first to get actual name
  void *handle = LoadDynamicLibrary(plugin_path);
  if (!handle) {
    return false;
  }

  auto handle_guard = [&handle, this]() {
    if (handle) {
      CloseDynamicLibrary(handle);
      handle = nullptr;
    }
  };

  auto create_func = GetCreatePluginFunction(handle, plugin_path);
  if (!create_func) {
    handle_guard();
    return false;
  }

  std::string actual_plugin_name;
  std::vector<std::string> available_passes;
  CustomPassPlugin *raw_plugin = nullptr;

  try {
    raw_plugin = create_func();
    if (!raw_plugin) {
      MS_LOG(ERROR) << "Failed to create plugin instance from: " << plugin_path;
      handle_guard();
      return false;
    }

    actual_plugin_name = raw_plugin->GetPluginName();

    try {
      available_passes = raw_plugin->GetAvailablePassNames();
    } catch (const std::exception &e) {
      available_passes.clear();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception creating plugin: " << e.what();
    if (raw_plugin) {
      try {
        DestroyPluginInstance(handle, raw_plugin);
      } catch (...) {
        delete raw_plugin;
      }
    }
    handle_guard();
    return false;
  }

  // Check if plugin already exists
  if (plugins_.find(actual_plugin_name) != plugins_.end()) {
    MS_LOG(DEBUG) << "Plugin " << actual_plugin_name << " already loaded, registering additional pass: " << pass_name;

    // Clean up the duplicate plugin
    try {
      DestroyPluginInstance(handle, raw_plugin);
    } catch (...) {
      delete raw_plugin;
    }
    handle_guard();  // Close the duplicate handle

    // Verify the pass is available in the existing plugin
    auto &existing_plugin_info = plugins_[actual_plugin_name];
    if (!ValidatePassExists(pass_name, actual_plugin_name, existing_plugin_info->available_passes)) {
      return false;  // Fail if pass doesn't exist
    }

    // Record pass registration with plugin association
    RegisterPassExecution(pass_name, actual_plugin_name, device, stage);
    return true;
  }

  // Verify the pass is available in this plugin
  if (!ValidatePassExists(pass_name, actual_plugin_name, available_passes)) {
    // Clean up the plugin
    try {
      DestroyPluginInstance(handle, raw_plugin);
    } catch (...) {
      delete raw_plugin;
    }
    handle_guard();
    return false;
  }

  // Create shared library handle wrapper for safe cleanup
  auto shared_handle = std::make_shared<LibraryHandle>(handle);

  // Convert raw plugin to shared_ptr with proper deleter
  auto plugin = std::shared_ptr<CustomPassPlugin>(raw_plugin, [shared_handle](CustomPassPlugin *p) {
    if (p && shared_handle->handle) {
      auto &manager = CustomPassPluginManager::GetInstance();
      manager.DestroyPluginInstance(shared_handle->handle, p);
    } else if (p) {
      delete p;
    }
  });

  // Create unified plugin info structure
  auto plugin_info = std::make_unique<PluginInfo>(actual_plugin_name, plugin_path, plugin, handle, shared_handle);
  plugin_info->available_passes = std::move(available_passes);

  // Store plugin info (RAII manages resources)
  plugins_[actual_plugin_name] = std::move(plugin_info);
  handle = nullptr;  // Transfer ownership to PluginInfo

  // Record pass registration with plugin association
  RegisterPassExecution(pass_name, actual_plugin_name, device, stage);

  MS_LOG(INFO) << "Successfully loaded plugin: " << actual_plugin_name << " from " << plugin_path
               << " with pass: " << pass_name;
  return true;
}

void CustomPassPluginManager::UnloadPlugin(const std::string &plugin_name) {
  if (plugins_.find(plugin_name) != plugins_.end()) {
    if (UnloadPluginInternal(plugin_name)) {
      MS_LOG(INFO) << "Successfully unloaded plugin: " << plugin_name;
    } else {
      MS_LOG(INFO) << "Plugin " << plugin_name << " is still in use and remains loaded";
    }
  }
}

void CustomPassPluginManager::UnloadAllPlugins() {
  // Copy plugin names to avoid iterator invalidation
  std::vector<std::string> plugin_names;
  plugin_names.reserve(plugins_.size());
  std::transform(plugins_.begin(), plugins_.end(), std::back_inserter(plugin_names),
                 [](const auto &pair) { return pair.first; });

  size_t initial_count = plugin_names.size();
  size_t unloaded_count = 0;
  size_t still_in_use_count = 0;
  std::unordered_set<std::string> unloaded_plugins;

  // Unload each plugin individually
  for (const auto &plugin_name : plugin_names) {
    try {
      if (UnloadPluginInternal(plugin_name)) {
        unloaded_count++;
        unloaded_plugins.insert(plugin_name);
        MS_LOG(INFO) << "Successfully unloaded plugin: " << plugin_name;
      } else {
        still_in_use_count++;
        MS_LOG(INFO) << "Plugin " << plugin_name << " is still in use and remains loaded";
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception during unload of plugin " << plugin_name << ": " << e.what();
      unloaded_count++;  // Count as unloaded since exception handling removes it
      unloaded_plugins.insert(plugin_name);
    }
  }

  // Only clear execution order entries for plugins that were actually unloaded
  // Keep entries for plugins that are still loaded
  if (!unloaded_plugins.empty()) {
    size_t original_size = pass_execution_order_.size();
    pass_execution_order_.erase(std::remove_if(pass_execution_order_.begin(), pass_execution_order_.end(),
                                               [&unloaded_plugins](const PassExecution &exec) {
                                                 return unloaded_plugins.find(exec.plugin_name) !=
                                                        unloaded_plugins.end();
                                               }),
                                pass_execution_order_.end());

    size_t removed_count = original_size - pass_execution_order_.size();
    if (removed_count > 0) {
      MS_LOG(INFO) << "Removed " << removed_count << " pass execution entries for unloaded plugins";
    }
  }

  // Report final status
  if (still_in_use_count == 0) {
    MS_LOG(INFO) << "All " << initial_count << " plugins unloaded successfully";
  } else {
    MS_LOG(INFO) << "Unloaded " << unloaded_count << " plugins, " << still_in_use_count
                 << " plugins remain loaded (still in use)";
  }
}

std::vector<std::shared_ptr<CustomPassPlugin>> CustomPassPluginManager::GetAllPlugins() {
  std::vector<std::shared_ptr<CustomPassPlugin>> result;
  result.reserve(plugins_.size());
  std::transform(plugins_.begin(), plugins_.end(), std::back_inserter(result),
                 [](const auto &pair) { return pair.second->plugin; });
  return result;
}

std::shared_ptr<CustomPassPlugin> CustomPassPluginManager::GetPlugin(const std::string &plugin_name) {
  auto iter = plugins_.find(plugin_name);
  if (iter != plugins_.end()) {
    return iter->second->plugin;
  }
  return nullptr;
}

void CustomPassPluginManager::RegisterPassesToOptimizer(std::shared_ptr<GraphOptimizer> optimizer,
                                                        const std::string &device) {
  MS_LOG(INFO) << "Starting pass registration in user-defined order for device: " << device;

  auto ordered_pass_manager = std::make_shared<PassManager>("custom_ordered_passes");

  for (const auto &pass_exec : pass_execution_order_) {
    // Check device compatibility using the recorded device for this specific call
    if (pass_exec.device != "all" && pass_exec.device != device) {
      MS_LOG(DEBUG) << "Skipping pass '" << pass_exec.pass_name << "' - device '" << pass_exec.device
                    << "' incompatible with target device '" << device << "'";
      continue;
    }

    // Find plugin by name directly (more efficient than searching by pass name)
    auto plugin_iter = plugins_.find(pass_exec.plugin_name);
    if (plugin_iter == plugins_.end() || !plugin_iter->second || !plugin_iter->second->plugin) {
      MS_LOG(WARNING) << "Plugin '" << pass_exec.plugin_name << "' for pass '" << pass_exec.pass_name
                      << "' not found or invalid";
      continue;
    }

    MS_EXCEPTION_IF_NULL(plugin_iter->second);
    // Create and add pass inline
    auto pass = plugin_iter->second->plugin->CreatePass(pass_exec.pass_name);
    if (pass == nullptr) {
      MS_LOG(ERROR) << "Failed to create pass '" << pass_exec.pass_name << "' from plugin '"
                    << plugin_iter->second->plugin->GetPluginName() << "'";
      continue;
    }

    ordered_pass_manager->AddPass(pass);
    MS_LOG(INFO) << "Added pass '" << pass_exec.pass_name << "' from plugin '"
                 << plugin_iter->second->plugin->GetPluginName() << "' in execution order position "
                 << ordered_pass_manager->Passes().size();
  }

  if (ordered_pass_manager->Passes().empty()) {
    MS_LOG(INFO) << "No passes registered for device: " << device;
    return;
  }

  MS_EXCEPTION_IF_NULL(optimizer);
  optimizer->AddPassManager(ordered_pass_manager);
  MS_LOG(INFO) << "Added ordered pass manager with " << ordered_pass_manager->Passes().size()
               << " passes for device: " << device;
}

// Helper function to validate if a pass exists in a plugin
bool CustomPassPluginManager::ValidatePassExists(const std::string &pass_name, const std::string &plugin_name,
                                                 const std::vector<std::string> &available_passes) {
  if (std::find(available_passes.begin(), available_passes.end(), pass_name) == available_passes.end()) {
    MS_LOG(ERROR) << "Pass '" << pass_name << "' not found in plugin '" << plugin_name
                  << "'. Available passes: " << FormatAvailablePassesList(available_passes);
    return false;
  }
  return true;
}

// Helper function to format available passes list for logging
std::string CustomPassPluginManager::FormatAvailablePassesList(const std::vector<std::string> &available_passes) {
  std::string result;
  for (const auto &pass : available_passes) {
    if (!result.empty()) result += ", ";
    result += pass;
  }
  return result;
}

// Helper function to register pass execution with consistent logging
void CustomPassPluginManager::RegisterPassExecution(const std::string &pass_name, const std::string &plugin_name,
                                                    const std::string &device, const std::string &stage) {
  pass_execution_order_.emplace_back(pass_name, plugin_name, device, stage, next_registration_order_++);
  MS_LOG(INFO) << "Added pass '" << pass_name << "' from plugin '" << plugin_name << "' with device '" << device
               << "' and stage '" << stage << "' to execution order at position " << next_registration_order_;
}

bool CustomPassPluginManager::IsPluginLoaded(const std::string &plugin_name) const {
  return plugins_.find(plugin_name) != plugins_.end();
}

std::vector<std::string> CustomPassPluginManager::GetLoadedPluginNames() const {
  std::vector<std::string> names;
  names.reserve(plugins_.size());
  std::transform(plugins_.begin(), plugins_.end(), std::back_inserter(names),
                 [](const auto &pair) { return pair.first; });
  return names;
}

bool CustomPassPluginManager::UnloadPluginInternal(const std::string &plugin_name) {
  auto plugin_iter = plugins_.find(plugin_name);
  if (plugin_iter == plugins_.end()) {
    MS_LOG(DEBUG) << "Plugin " << plugin_name << " not found during unload";
    return false;
  }

  try {
    auto &plugin_info = plugin_iter->second;

    // Check if plugin is still in use by external code
    if (plugin_info->plugin && plugin_info->plugin.use_count() > 1) {
      MS_LOG(DEBUG) << "Plugin " << plugin_name << " is still in use (ref count: " << plugin_info->plugin.use_count()
                    << "). Cannot unload safely.";
      return false;
    }
    // Remove all pass registrations for this plugin first
    size_t original_size = pass_execution_order_.size();
    pass_execution_order_.erase(
      std::remove_if(pass_execution_order_.begin(), pass_execution_order_.end(),
                     [&plugin_name](const PassExecution &exec) { return exec.plugin_name == plugin_name; }),
      pass_execution_order_.end());

    size_t removed_count = original_size - pass_execution_order_.size();
    if (removed_count > 0) {
      MS_LOG(INFO) << "Removed " << removed_count
                   << " pass registrations from execution order for plugin: " << plugin_name;
    }

    // Safe to remove plugin from container - no external references exist
    plugins_.erase(plugin_iter);
    return true;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception during plugin unload for " << plugin_name << ": " << e.what();
    // Ensure cleanup even in exceptional cases
    plugins_.erase(plugin_name);

    // Fallback cleanup for pass registrations
    pass_execution_order_.erase(
      std::remove_if(pass_execution_order_.begin(), pass_execution_order_.end(),
                     [&plugin_name](const PassExecution &exec) { return exec.plugin_name == plugin_name; }),
      pass_execution_order_.end());
    return true;  // Force unloaded despite exception
  }
}

void *CustomPassPluginManager::LoadDynamicLibrary(const std::string &plugin_path) {
#if !defined(_WIN32) && !defined(_WIN64)
  void *handle = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MS_LOG(ERROR) << "Failed to load plugin: " << plugin_path << ", error: " << dlerror();
    return nullptr;
  }
#else
  void *handle = LoadLibrary(plugin_path.c_str());
  if (!handle) {
    MS_LOG(ERROR) << "Failed to load plugin: " << plugin_path << ", error: " << GetLastError();
    return nullptr;
  }
#endif
  return handle;
}

CustomPassPluginManager::CreatePluginFunc CustomPassPluginManager::GetCreatePluginFunction(
  void *handle, const std::string &plugin_path) {
#if !defined(_WIN32) && !defined(_WIN64)
  auto create_func = reinterpret_cast<CreatePluginFunc>(dlsym(handle, "CreatePlugin"));
  if (!create_func) {
    MS_LOG(ERROR) << "Failed to find CreatePlugin function in: " << plugin_path << ", error: " << dlerror();
    return nullptr;
  }
#else
  auto create_func = reinterpret_cast<CreatePluginFunc>(GetProcAddress(static_cast<HMODULE>(handle), "CreatePlugin"));
  if (!create_func) {
    MS_LOG(ERROR) << "Failed to find CreatePlugin function in: " << plugin_path << ", error: " << GetLastError();
    return nullptr;
  }
#endif
  return create_func;
}

void CustomPassPluginManager::DestroyPluginInstance(void *handle, CustomPassPlugin *plugin) {
  if (!plugin) return;

#if !defined(_WIN32) && !defined(_WIN64)
  auto destroy_func = reinterpret_cast<DestroyPluginFunc>(dlsym(handle, "DestroyPlugin"));
  if (destroy_func) {
    destroy_func(plugin);
  } else {
    // Fallback to standard delete if DestroyPlugin function not found
    delete plugin;
  }
#else
  auto destroy_func =
    reinterpret_cast<DestroyPluginFunc>(GetProcAddress(static_cast<HMODULE>(handle), "DestroyPlugin"));
  if (destroy_func) {
    destroy_func(plugin);
  } else {
    // Fallback to standard delete if DestroyPlugin function not found
    delete plugin;
  }
#endif
}

void CustomPassPluginManager::CloseDynamicLibrary(void *handle) {
#if !defined(_WIN32) && !defined(_WIN64)
  dlclose(handle);
#else
  FreeLibrary(static_cast<HMODULE>(handle));
#endif
}

void CustomPassPluginManager::PluginInfo::CleanupResources() {
  // DO NOT manually destroy plugin or close library handle here
  // The shared_ptr's custom deleter handles plugin destruction
  // The shared LibraryHandle's destructor handles library cleanup
  // Both will happen automatically when reference counts reach zero

  // Just clear our local references - this may trigger cleanup if we hold the last references
  plugin.reset();
  shared_handle.reset();
  handle = nullptr;
}

}  // namespace opt
}  // namespace mindspore
