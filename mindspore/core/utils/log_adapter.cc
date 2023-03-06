/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "utils/log_adapter.h"

#ifndef _MSC_VER
#include <unistd.h>
#include <sys/time.h>
#endif
#include <map>
#include <iomanip>
#include <regex>
#include <thread>
#include <vector>
#include "utils/convert_utils_base.h"

// namespace to support utils module definition
namespace mindspore {
constexpr int kNameMaxLength = 18;
constexpr size_t kStep = 2;
constexpr auto kSplitLine = "\n----------------------------------------------------\n";
constexpr auto kFrameworkErrorTitle = "Framework Error Message:";
// set default log level to WARNING for all sub modules
int g_ms_submodule_log_levels[NUM_SUBMODUES] = {MsLogLevel::kWarning};
#if defined(_WIN32) || defined(_WIN64)
enum MsLogLevel this_thread_max_log_level = MsLogLevel::kException;
#else
thread_local enum MsLogLevel this_thread_max_log_level = MsLogLevel::kException;
#endif

#ifdef USE_GLOG
#define google mindspore_private
static std::string GetProcName() {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const std::string appname = getprogname();
#elif defined(_GNU_SOURCE)
  const std::string appname = program_invocation_name;
#else
  const std::string appname = "?";
#endif
  // sometimes, the app name is an absolute path, it is too long
  std::string app_name(appname);
  std::size_t pos = app_name.rfind("/");
  if (pos == std::string::npos) {
    return app_name;
  }
  if (pos + 1 >= app_name.size()) {
    return app_name;
  }
  return app_name.substr(pos + 1);
}

static std::string GetLogLevel(MsLogLevel level) {
  static const char *const level_names[] = {
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
  };
  if (level > this_thread_max_log_level) {
    level = this_thread_max_log_level;
  }
  return std::string(level_names[level]);
}

// convert MsLogLevel to corresponding glog level
static int GetGlogLevel(MsLogLevel level) {
  switch (level >= this_thread_max_log_level ? this_thread_max_log_level : level) {
    case MsLogLevel::kDebug:
    case MsLogLevel::kInfo:
      return google::GLOG_INFO;
    case MsLogLevel::kWarning:
      return google::GLOG_WARNING;
    case MsLogLevel::kError:
    case MsLogLevel::kException:
    default:
      return google::GLOG_ERROR;
  }
}

// get threshold level
static int GetThresholdLevel(const std::string &threshold) {
  if (threshold.empty()) {
    return google::GLOG_WARNING;
  } else if (threshold == "DEBUG" || threshold == "INFO") {
    return google::GLOG_INFO;
  } else if (threshold == "WARNING") {
    return google::GLOG_WARNING;
  } else if (threshold == "ERROR" || threshold == "CRITICAL") {
    return google::GLOG_ERROR;
  } else {
    return google::GLOG_WARNING;
  }
}
#undef google
#else

#undef Dlog
#define Dlog(module_id, level, format, ...)                   \
  do {                                                        \
    DlogInner((module_id), (level), (format), ##__VA_ARGS__); \
  } while (0)

// convert MsLogLevel to corresponding slog level
static int GetSlogLevel(MsLogLevel level) {
  switch (level) {
    case MsLogLevel::kDebug:
      return DLOG_DEBUG;
    case MsLogLevel::kInfo:
      return DLOG_INFO;
    case MsLogLevel::kWarning:
      return DLOG_WARN;
    case MsLogLevel::kError:
    case MsLogLevel::kException:
    default:
      return DLOG_ERROR;
  }
}
#endif

LogWriter::ExceptionHandler &LogWriter::exception_handler() {
  static LogWriter::ExceptionHandler g_exception_handler = nullptr;
  return g_exception_handler;
}

LogWriter::MessageHandler &LogWriter::message_handler() {
  static LogWriter::MessageHandler g_message_handler = nullptr;
  return g_message_handler;
}

LogWriter::TraceProvider &LogWriter::trace_provider() {
  static LogWriter::TraceProvider g_trace_provider = nullptr;
  return g_trace_provider;
}

const LogWriter::ExceptionHandler &LogWriter::GetExceptionHandler() {
  const auto &exception_handler_tmp = exception_handler();
  return exception_handler_tmp;
}

void LogWriter::SetExceptionHandler(const LogWriter::ExceptionHandler &new_exception_handler) {
  auto &exception_handler_tmp = exception_handler();
  exception_handler_tmp = new_exception_handler;
}

const LogWriter::MessageHandler &LogWriter::GetMessageHandler() { return message_handler(); }

void LogWriter::SetMessageHandler(const LogWriter::MessageHandler &new_message_handler) {
  message_handler() = new_message_handler;
}

const LogWriter::TraceProvider &LogWriter::GetTraceProvider() {
  const auto &trace_provider_tmp = trace_provider();
  return trace_provider_tmp;
}

void LogWriter::SetTraceProvider(const LogWriter::TraceProvider &new_trace_provider) {
  auto &trace_provider_tmp = trace_provider();
  if (trace_provider_tmp != nullptr) {
    MS_LOG(INFO) << "trace provider has been set, skip.";
    return;
  }
  trace_provider_tmp = new_trace_provider;
}

static inline std::string GetEnv(const std::string &envvar) {
  const char *value = std::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

// When GLOG_logtostderr is set to 0, logs are output to a file, then will print duplicate message
// in exception log and stack. Otherwise when GLOG_logtostderr is set to 1, logs are output to the screen,
// then will only print message in exception stack.
static MsLogLevel GetGlobalLogLevel();
void LogWriter::RemoveLabelBeforeOutputLog(const std::ostringstream &msg) const {
  auto logLevel = GetGlobalLogLevel();
  if (logLevel <= MsLogLevel::kInfo || GetEnv("GLOG_logtostderr") == "0") {
    std::string str = msg.str();
    // remove any titles enclosed in "#dmsg#" or "#umsg#", as well as its formatted couterparts
    std::regex title_re{R"(\#[d|u]msg\#.+?\#[d|u]msg\#|)" + std::string(kSplitLine) + R"(- .+?)" +
                        std::string(kSplitLine)};
    std::ostringstream replaced_msg;
    replaced_msg << std::regex_replace(str, title_re, "");
    OutputLog(replaced_msg);
  }
}

// Function to split string based on character delimiter
void SplitString(const std::string &message, const std::string &delimiter, std::vector<std::string> *output) {
  size_t pos1, pos2;
  pos1 = 0;
  pos2 = message.find(delimiter);
  MS_EXCEPTION_IF_NULL(output);

  while (pos2 != std::string::npos) {
    (void)output->emplace_back(message.substr(pos1, pos2 - pos1));
    pos1 = pos2 + delimiter.size();
    pos2 = message.find(delimiter, pos1);
  }

  if (pos1 != message.length()) {
    (void)output->emplace_back(message.substr(pos1));
  }
}

// Parse exception message format like: Error Description#dmsg#Developer Message Title#dmsg#Developer Message Content
// #umsg#User Message Title#umsg#User Message Content
void ParseExceptionMessage(const std::string &message, std::ostringstream &oss, std::vector<std::string> *dmsg,
                           std::vector<std::string> *umsg) {
  std::vector<std::string> vec;
  SplitString(message, "#dmsg#", &vec);  // first:split message by label #dmsg#

  if (!vec.empty() && (vec[0].find("#umsg#") == std::string::npos)) {
    oss << vec[0];
  }

  MS_EXCEPTION_IF_NULL(dmsg);
  MS_EXCEPTION_IF_NULL(umsg);

  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].find("#umsg#") != std::string::npos) {
      std::vector<std::string> temp;
      SplitString(vec[i], "#umsg#", &temp);  // second:split message by label #umsg#
      if (!temp.empty()) {
        if (i == 0) {
          oss << temp[0];
        } else {
          (void)dmsg->emplace_back(temp[0]);
        }
        (void)umsg->insert(umsg->cend(), temp.cbegin() + 1, temp.cend());
      }
    } else {
      if (i != 0) {
        (void)dmsg->emplace_back(vec[i]);
      }
    }
  }
}

void PrintMessage(std::ostringstream &oss, const std::string &title, const std::string &content) {
  const std::string &message = oss.str();
  size_t length = message.length();
  if ((length != 0) && (message[length - 1] != '\n')) {
    oss << "\n";
  }

  oss << kSplitLine << title << kSplitLine << content;
}

void CombineExceptionMessageWithSameTitle(std::ostringstream &oss, const std::string &title,
                                          const std::string &content) {
  if (title.empty() || content.empty()) {
    return;
  }
  // ignore the passed framework title if content has a formatted title itself
  if (title.find(kFrameworkErrorTitle) != std::string::npos && content.find(kSplitLine) == 0) {
    oss << '\n' << content;
    return;
  }
  std::string message = oss.str();
  size_t position = message.find(title + kSplitLine);
  if (position != std::string::npos) {
    position += title.length() + strlen(kSplitLine);
    if (content[content.length() - 1] != '\n') {
      (void)message.insert(position, content + "\n");
    } else {
      (void)message.insert(position, content);
    }
    oss.str("");
    oss << message;
  } else {
    PrintMessage(oss, title, content);
  }
}

void DisplayDevExceptionMessage(std::ostringstream &oss, const std::vector<std::string> &dmsg,
                                const LocationInfo &location) {
  bool display = true;
  if (GetEnv("MS_EXCEPTION_DISPLAY_LEVEL") == "1") {
    display = false;
  }

  if (display) {
    size_t size = dmsg.size();
    if ((size != 0) && (size % kStep == 0)) {
      for (size_t i = 0; i < size; i += kStep) {
        std::ostringstream dmsg_title;
        dmsg_title << "- " << dmsg[i] << " (For framework developers)";
        CombineExceptionMessageWithSameTitle(oss, dmsg_title.str(), dmsg[i + 1]);
      }
    }

    const std::string CPP_CALL_STACK_TITLE = "- C++ Call Stack: (For framework developers)";
    std::ostringstream cpp_call_stack_content;
    cpp_call_stack_content << location.file_ << ":" << location.line_ << " " << location.func_ << "\n";
    CombineExceptionMessageWithSameTitle(oss, CPP_CALL_STACK_TITLE, cpp_call_stack_content.str());
  }
}

void DisplayUserExceptionMessage(std::ostringstream &oss, const std::vector<std::string> &umsg) {
  size_t size = umsg.size();
  if ((size != 0) && (size % kStep == 0)) {
    for (size_t i = 0; i < size; i += kStep) {
      std::ostringstream umsg_title;
      umsg_title << "- " << umsg[i];
      CombineExceptionMessageWithSameTitle(oss, umsg_title.str(), umsg[i + 1]);
    }
  }
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
#ifdef USE_GLOG
#define google mindspore_private
  auto submodule_name = GetSubModuleName(submodule_);
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
#ifdef _MSC_VER
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "("
    << "," << std::hex
#else
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "(" << getpid() << "," << std::hex
#endif
    << std::this_thread::get_id() << std::dec << "," << GetProcName() << "):" << GetTimeString() << " "
    << "[" << location_.file_ << ":" << location_.line_ << "] " << location_.func_ << "] " << msg.str() << std::endl;
#undef google
#else
  auto str_msg = msg.str();
  auto slog_module_id = (submodule_ == SM_MD ? MD : ME);
  Dlog(static_cast<int>(slog_module_id), GetSlogLevel(log_level_), "[%s:%d] %s] %s", location_.file_, location_.line_,
       location_.func_, str_msg.c_str());
#endif
}

void LogWriter::operator<(const LogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
void LogWriter::operator^(const LogStream &stream) const {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();

  const auto &message_handler = GetMessageHandler();
  if (message_handler != nullptr) {
    message_handler(&msg);
  }

  std::ostringstream oss;
  std::vector<std::string> dmsg;
  std::vector<std::string> umsg;

  ParseExceptionMessage(msg.str(), oss, &dmsg, &umsg);
  DisplayUserExceptionMessage(oss, umsg);

  thread_local bool running = false;
  if (!running) {
    running = true;
    if (this_thread_max_log_level >= MsLogLevel::kException) {
      RemoveLabelBeforeOutputLog(msg);
    }
    const auto &trace_provider = GetTraceProvider();
    if (trace_provider != nullptr) {
      trace_provider(oss, true);
    }
    running = false;
  }

  DisplayDevExceptionMessage(oss, dmsg, location_);

  const auto &exception_handler = GetExceptionHandler();
  if (exception_handler != nullptr) {
    exception_handler(exception_type_, oss.str());
  } else {
    oss << "[Runtime error for null exception handler]";
    throw std::runtime_error(oss.str());
  }
  oss << "[Runtime error for unknown exception type:" << exception_type_ << "]";
  throw std::runtime_error(oss.str());
}

enum class LogConfigToken : size_t {
  INVALID,      // indicate invalid token
  LEFT_BRACE,   // '{'
  RIGHT_BRACE,  // '}'
  VARIABLE,     // '[A-Za-z][A-Za-z0-9_]*'
  NUMBER,       // [0-9]+
  COMMA,        // ','
  COLON,        // ':'
  EOS,          // End Of String, '\0'
  NUM_LOG_CFG_TOKENS
};

static const char *g_tok_names[static_cast<size_t>(LogConfigToken::NUM_LOG_CFG_TOKENS)] = {
  "invalid",        // indicate invalid token
  "{",              // '{'
  "}",              // '}'
  "variable",       // '[A-Za-z][A-Za-z0-9_]*'
  "number",         // [0-9]+
  ",",              // ','
  ":",              // ':'
  "end-of-string",  // End Of String, '\0'
};

static inline bool IsAlpha(char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }

static inline bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

class LogConfigLexer {
 public:
  explicit LogConfigLexer(const std::string &text) : buffer_(text), cur_idx_(0) {}
  ~LogConfigLexer() = default;

  // skip white space, and return the first char after white space
  char SkipWhiteSpace() {
    while (cur_idx_ < buffer_.size()) {
      char ch = buffer_[cur_idx_];
      if (ch == ' ' || ch == '\t') {
        ++cur_idx_;
        continue;
      }
      return ch;
    }
    return '\0';
  }

  LogConfigToken GetNext(std::string *const ptr) {
#ifdef DEBUG
    std::string text;
    auto tok = GetNextInner(&text);
    MS_LOG(DEBUG) << "Got token " << tok << " with value [" << text << "]";
    if (ptr != nullptr) {
      *ptr = text;
    }
    return tok;
  }

  LogConfigToken GetNextInner(std::string *ptr) {
#endif
    char ch = SkipWhiteSpace();
    // clang-format off
    static const std::map<char, LogConfigToken> single_char_map = {
      {'{', LogConfigToken::LEFT_BRACE},
      {'}', LogConfigToken::RIGHT_BRACE},
      {',', LogConfigToken::COMMA},
      {':', LogConfigToken::COLON},
      {'\0', LogConfigToken::EOS},
    };
    // clang-format on

    auto iter = single_char_map.find(ch);
    if (iter != single_char_map.end()) {
      if (ptr != nullptr) {
        *ptr = std::string() + ch;
      }
      ++cur_idx_;
      return iter->second;
    } else if (IsAlpha(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && (IsAlpha(ch) || IsDigit(ch) || ch == '_'));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::VARIABLE;
    } else if (IsDigit(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && IsDigit(ch));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::NUMBER;
    }
    return LogConfigToken::INVALID;
  }

 private:
  std::string buffer_;
  size_t cur_idx_;
};

class LogConfigParser {
 public:
  explicit LogConfigParser(const std::string &cfg) : lexer(cfg) {}
  ~LogConfigParser() = default;

  bool Expect(LogConfigToken expected, LogConfigToken tok) const {
    if (expected != tok) {
      MS_LOG(WARNING) << "Parse submodule log configuration text error, expect `"
                      << g_tok_names[static_cast<size_t>(expected)] << "`, but got `"
                      << g_tok_names[static_cast<size_t>(tok)] << "`. The whole configuration will be ignored.";
      return false;
    }
    return true;
  }

  // The text of config MS_SUBMODULE_LOG_v is in the form {submodule1:log_level1,submodule2:log_level2,...}.
  // Valid values of log levels are: 0 - debug, 1 - info, 2 - warning, 3 - error, 4 - critical
  // e.g. MS_SUBMODULE_LOG_v={PARSER:0, ANALYZER:2, PIPELINE:1}
  std::map<std::string, std::string> Parse() {
    std::map<std::string, std::string> log_levels;

    bool flag_error = false;
    std::string text;
    auto tok = lexer.GetNext(&text);
    // empty string
    if (tok == LogConfigToken::EOS) {
      return log_levels;
    }

    if (!Expect(LogConfigToken::LEFT_BRACE, tok)) {
      return log_levels;
    }

    do {
      std::string key, val;
      tok = lexer.GetNext(&key);
      if (!Expect(LogConfigToken::VARIABLE, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&text);
      if (!Expect(LogConfigToken::COLON, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&val);
      if (!Expect(LogConfigToken::NUMBER, tok)) {
        flag_error = true;
        break;
      }

      log_levels[key] = val;
      tok = lexer.GetNext(&text);
    } while (tok == LogConfigToken::COMMA);

    if (!flag_error && !Expect(LogConfigToken::RIGHT_BRACE, tok)) {
      flag_error = true;
    }

    if (flag_error) {
      log_levels.clear();
    }
    return log_levels;
  }

 private:
  LogConfigLexer lexer;
};

bool ParseLogLevel(const std::string &str_level, MsLogLevel *ptr_level) {
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    constexpr char number_start = '0';
    ch = ch - number_start;  // subtract ASCII code of '0', which is 48
    if (ch >= static_cast<int>(MsLogLevel::kDebug) && ch <= static_cast<int>(MsLogLevel::kException)) {
      if (ptr_level != nullptr) {
        *ptr_level = static_cast<MsLogLevel>(ch);
      }
      return true;
    }
  }
  return false;
}

static MsLogLevel GetGlobalLogLevel() {
#ifdef USE_GLOG
  return static_cast<MsLogLevel>(FLAGS_v);
#else
  int log_level = MsLogLevel::kWarning;  // set default log level to WARNING
  auto str_level = GetEnv("GLOG_v");
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    constexpr char number_start = '0';
    ch = ch - number_start;  // subtract ASCII code of '0', which is 48
    if (ch >= MsLogLevel::kDebug && ch <= MsLogLevel::kException) {
      log_level = ch;
    }
  }
  return static_cast<MsLogLevel>(log_level);
#endif
}

void InitSubModulesLogLevel() {
  // initialize submodule's log level using global
  auto global_log_level = GetGlobalLogLevel();
  for (int i = 0; i < static_cast<int>(NUM_SUBMODUES); ++i) {
    g_ms_submodule_log_levels[i] = global_log_level;
  }

  // set submodule's log level
  auto submodule = GetEnv("MS_SUBMODULE_LOG_v");
  MS_LOG(DEBUG) << "MS_SUBMODULE_LOG_v=`" << submodule << "`";
  LogConfigParser parser(submodule);
  auto configs = parser.Parse();
  for (const auto &cfg : configs) {
    int mod_idx = -1;
    for (int i = 0; i < static_cast<int>(NUM_SUBMODUES); ++i) {
      if (cfg.first == GetSubModuleName(static_cast<SubModuleId>(i))) {
        mod_idx = i;
        break;
      }
    }
    if (mod_idx < 0) {
      MS_LOG(WARNING) << "Undefined module name " << cfg.first << ", ignore it";
      continue;
    }
    MsLogLevel submodule_log_level;
    if (!ParseLogLevel(cfg.second, &submodule_log_level)) {
      MS_LOG(WARNING) << "Illegal log level value " << cfg.second << " for " << cfg.first << ", ignore it.";
      continue;
    }
    g_ms_submodule_log_levels[mod_idx] = static_cast<int>(submodule_log_level);
  }
}

const std::string GetSubModuleName(SubModuleId module_id) {
  static const char sub_module_names[NUM_SUBMODUES][kNameMaxLength] = {
    "UNKNOWN",            // SM_UNKNOWN
    "CORE",               // SM_CORE
    "ANALYZER",           // SM_ANALYZER
    "COMMON",             // SM_COMMON
    "DEBUG",              // SM_DEBUG
    "OFFLINE_DEBUG",      // SM_OFFLINE_DEBUG
    "DEVICE",             // SM_DEVICE
    "GE_ADPT",            // SM_GE_ADPT
    "IR",                 // SM_IR
    "KERNEL",             // SM_KERNEL
    "MD",                 // SM_MD
    "ME",                 // SM_ME
    "EXPRESS",            // SM_EXPRESS
    "OPTIMIZER",          // SM_OPTIMIZER
    "PARALLEL",           // SM_PARALLEL
    "PARSER",             // SM_PARSER
    "PIPELINE",           // SM_PIPELINE
    "PRE_ACT",            // SM_PRE_ACT
    "PYNATIVE",           // SM_PYNATIVE
    "SESSION",            // SM_SESSION
    "UTILS",              // SM_UTILS
    "VM",                 // SM_VM
    "PROFILER",           // SM_PROFILER
    "PS",                 // SM_PS
    "FL",                 // SM_FL
    "DISTRIBUTED",        // SM_DISTRIBUTED
    "LITE",               // SM_LITE
    "ARMOUR",             // SM_ARMOUR
    "HCCL_ADPT",          // SM_HCCL_ADPT
    "RUNTIME_FRAMEWORK",  // SM_RUNTIME_FRAMEWORK
    "GE",                 // SM_GE
    "API",                // SM_API
  };
  return sub_module_names[IntToSize(module_id % NUM_SUBMODUES)];
}

std::string GetTimeString() {
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  constexpr int base_year = 1900;
  std::stringstream ss;
  ss << now_time.tm_year + base_year << "-" << now_time.tm_mon + 1 << "-" << now_time.tm_mday << " " << now_time.tm_hour
     << ":" << now_time.tm_min << ":" << now_time.tm_sec;
  return ss.str();
#else
  constexpr auto BUFLEN = 80;
  char buf[BUFLEN] = {'\0'};
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  constexpr int width = 3;
  constexpr int64_t time_convert_unit = 1000;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  std::stringstream ss;
  ss << "." << std::setfill('0') << std::setw(width) << cur_time.tv_usec / time_convert_unit << "." << std::setfill('0')
     << std::setw(width) << cur_time.tv_usec % time_convert_unit;
  return std::string(buf) + ss.str();
#endif
}
}  // namespace mindspore

extern "C" {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#ifdef _MSC_VER
MS_CORE_API void common_log_init(void) {
#else
__attribute__((constructor)) MS_CORE_API void common_log_init(void) {
#endif
#else
MS_CORE_API void common_log_init(void) {
#endif
#ifdef USE_GLOG
  // Do not use glog predefined log prefix
  FLAGS_log_prefix = false;
  // Write log to files real-time
  FLAGS_logbufsecs = 0;
  // Set default log level to WARNING
  if (mindspore::GetEnv("GLOG_v").empty()) {
    FLAGS_v = static_cast<int>(mindspore::MsLogLevel::kWarning);
  }

  // Set default log file mode to 0640
  if (mindspore::GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  // Set default log file max size to 50 MB
  FLAGS_max_log_size = 50;
  std::string max_log_size = mindspore::GetEnv("GLOG_max_log_size");
  if (!max_log_size.empty()) {
    FLAGS_max_log_size = std::stoi(max_log_size);
  }
  std::string logtostderr = mindspore::GetEnv("GLOG_logtostderr");
  // Default print log to screen
  FLAGS_logtostderr = true;
  if (logtostderr == "0") {
    if (mindspore::GetEnv("GLOG_log_dir").empty()) {
      MS_LOG(ERROR) << "`GLOG_log_dir` is empty, it must be set while 'logtostderr' equals to 0.";
      // Here can not throw exception and use python to catch, because the PYBIND11_MODULE is not yet been initialed.
      exit(EXIT_FAILURE);
    } else {
      FLAGS_logtostderr = false;
      // Set log dir from GLOG_log_dir with RANK_ID or OMPI_COMM_WORLD_RANK.

      const std::string rank_id = mindspore::GetEnv("RANK_ID");
      const std::string gpu_rank_id = mindspore::GetEnv("OMPI_COMM_WORLD_RANK");
      std::string rank = "0";
      if (!rank_id.empty()) {
        rank = rank_id;
      } else if (!gpu_rank_id.empty()) {
        rank = gpu_rank_id;
      }
      FLAGS_log_dir = mindspore::GetEnv("GLOG_log_dir") + "/rank_" + rank + "/logs";
    }
  }

  // Default GLOG_stderrthreshold level to WARNING
  auto threshold = mindspore::GetEnv("GLOG_stderrthreshold");
  FLAGS_stderrthreshold = mindspore::GetThresholdLevel(threshold);

#endif
  mindspore::InitSubModulesLogLevel();
}

// shared lib init hook
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#if defined(_MSC_VER)
MS_CORE_API void mindspore_log_init(void) {
#else
__attribute__((constructor)) MS_CORE_API void mindspore_log_init(void) {
#endif
#else
MS_CORE_API void mindspore_log_init(void) {
#endif
#ifdef USE_GLOG
#define google mindspore_private
  static bool is_glog_initialzed = false;
  if (!is_glog_initialzed) {
    google::InitGoogleLogging("mindspore");
    is_glog_initialzed = true;
  }
#undef google
#endif
  common_log_init();
}

#ifdef _MSC_VER
typedef void(__cdecl *PF)(void);
#pragma section(".CRT$XCG", read)
__declspec(allocate(".CRT$XCG")) PF f[] = {mindspore_log_init};
#endif
}
