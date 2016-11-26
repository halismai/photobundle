#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <execinfo.h>

#include "debug.h"
#include "utils.h"

#include <glob.h>

#include <algorithm>

#include <cstdarg>
#include <thread>
#include <ctime>
#include <chrono>
#include <sstream>
#include <vector>
#include <fstream>

#if defined(WITH_BOOST)
#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/algorithm/string.hpp>
#endif

namespace utils {

using std::string;
using std::vector;

namespace po = boost::program_options;

ProgramOptions::ProgramOptions(std::string name)
     : _desc(name) { addOption("help,h", "Print help message and exit"); }

void ProgramOptions::parse(int argc, char** argv)
{
  try {
    po::store(po::parse_command_line(argc, argv, _desc), _vm);
  } catch(const std::exception& ex) {
    std::cerr << "error: " << ex.what() << std::endl;
    throw ex;
  }

  po::notify(_vm);

  if(hasOption("help"))
    printHelpAndExit();
}

bool ProgramOptions::hasOption(std::string name) const
{
  return _vm.count(name);
}

void ProgramOptions::printHelpAndExit(int exit_code) const
{
  std::cout << _desc << std::endl;
  exit(exit_code);
}

ProgramOptions& ProgramOptions::addOption(std::string name, std::string help)
{
  _desc.add_options()(name.c_str(), help.c_str());
  return *this;
}

std::vector<std::string> glob(const std::string& pattern, bool verbose)
{
  std::vector<std::string> ret;

  ::glob_t globbuf;
  int err = ::glob(pattern.c_str(), GLOB_TILDE, NULL, &globbuf);
  switch(err)
  {
    case GLOB_NOSPACE:
      {
        if(verbose) Warn("glob(): out of  memory\n");
        break;
      }
    case GLOB_ABORTED:
      {
        if(verbose) Warn("glob() : aborted. read error\n");
        break;
      }
    case GLOB_NOMATCH :
      {
        if(verbose) Warn("glob() : no match for: '%s'\n", pattern.c_str());
        break;
    }
  }

  if(!err)
  {
    const int count = globbuf.gl_pathc;
    ret.resize(count);
    for(int i = 0; i < count; ++i)
      ret[i] = std::string(globbuf.gl_pathv[i]);
  }

  globfree(&globbuf);
  return ret;
}

std::string Format(const char* fmt, ...)
{
  std::vector<char> buf(1024);

  while(true) {
    va_list va;
    va_start(va, fmt);
    auto len = vsnprintf(buf.data(), buf.size(), fmt, va);
    va_end(va);

    if(len < 0 || len >= (int) buf.size()) {
      buf.resize(std::max((int)(buf.size() << 1), len + 1));
      continue;
    }

    return std::string(buf.data(), len);
  }
}

void Sleep(int32_t ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


ConfigFile::ConfigFile() {}

ConfigFile::ConfigFile(std::ifstream& ifs)
{
  if(!ifs.is_open())
    throw Error("input file is not open");

  parse(ifs);
}

ConfigFile::ConfigFile(std::string filename)
{
  std::ifstream ifs(filename);
  if(!ifs.is_open())
    throw Error("could not open file '" + filename + "'");

  parse(ifs);
}

void ConfigFile::parse(std::ifstream& ifs)
{
  std::string line;
  while(!ifs.eof()) {
    std::getline(ifs, line);

    if(line.empty())
      continue;

    if(line.front() == '#' || line.front() == '%')
      continue;

    line.erase(std::remove_if(std::begin(line), std::end(line),
            [](char c) { return std::isspace(c); } ), std::end(line));

    const auto tokens = splitstr(line, '=');
    if(tokens.size() != 2)
      throw Error("Malformed ConfigFile line " + line);

    _data[tokens[0]] = tokens[1];
  }
}

ConfigFile& ConfigFile::operator()(const std::string& key,
                                   const std::string& value)
{
  _data[key] = value;
  return *this;
}

std::ostream& operator<<(std::ostream& os, const ConfigFile& cf)
{
  for(const auto& it : cf._data) {
    os << it.first << " = " << it.second << std::endl;
  }

  return os;
}

bool ConfigFile::save(std::string filename) const
{
  std::ofstream ofs(filename);
  if(ofs.is_open())
    ofs << *this;
  return !ofs.bad();
}

struct NoCaseCmp {
  inline bool operator()(const unsigned char& c1,
                         const unsigned char& c2) const
  {
    return std::tolower(c1) < std::tolower(c2);
  }
}; // NoCaseCmp

bool CaseInsenstiveComparator::operator()(const std::string& a,
                                          const std::string& b) const
{
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                      NoCaseCmp());
}

bool icompare(const std::string& a, const std::string& b)
{
#if defined(WITH_BOOST)
  return boost::algorithm::iequals(a, b);
#else
  return a.size() == b.size() ? !strncasecmp(a.c_str(), b.c_str(), a.size()) : false;
#endif
}


template<> int str2num<int>(const std::string& s) { return std::stoi(s); }

template <> double str2num<double>(const std::string& s) { return std::stod(s); }

template <> float str2num<float>(const std::string& s) { return std::stof(s); }

template <> bool str2num<bool>(const std::string& s)
{
  if(icompare(s, "true")) {
    return true;
  } else if(icompare(s, "false")) {
    return false;
  } else {
    // try to parse a bool from int {0,1}
    int v = str2num<int>(s);
    if(v == 0)
      return false;
    else if(v == 1)
      return true;
    else
      throw std::invalid_argument("string is not a boolean");
  }
}

vector<string> splitstr(const string& str, char delim)
{
  vector<string> ret;
  std::stringstream ss(str);
  string token;
  while(std::getline(ss, token, delim))
    ret.push_back(token);

  return ret;
}

namespace fs {

string expand_tilde(string fn)
{
  if(fn.front() == '~') {
    string home = getenv("HOME");
    if(home.empty()) {
      std::cerr << "could not query $HOME\n";
      return fn;
    }

    // handle the case when name == '~' only
    return home + dirsep(home) + ((fn.length()==1) ? "" :
                                  fn.substr(1,std::string::npos));
  } else {
    return fn;
  }
}

string dirsep(string dname)
{
  return (dname.back() == '/') ? "" : "/";
}

string extension(string filename)
{
#if defined(WITH_BOOST)
  return boost::filesystem::extension(filename);
#else
  auto i = filename.find_last_of(".");
  return (string::npos != i) ? filename.substr(i) : "";
#endif
}

bool exists(string path)
{
#if defined(WITH_BOOST)
  return boost::filesystem::exists(path);
#else
  struct stat buf;
  return (0 == stat(path.c_str(), &buf));
#endif
}

bool is_regular(string path)
{
#if defined(WITH_BOOST)
  return boost::filesystem::is_regular_file(path);
#else
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISREG(buf.st_mode) : false;
#endif
}

bool is_dir(string path)
{
#if defined(WITH_BOOST)
  return boost::filesystem::is_directory(path);
#else
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISDIR(buf.st_mode) : false;
#endif
}

bool try_make_dir(string dname, int mode = 0777)
{
  return (0 == ::mkdir(dname.c_str(), mode));
}

string mkdir(string dname, bool try_unique, int max_tries)
{
  if(!try_unique) {
    return try_make_dir(dname.c_str()) ? dname : "";
  } else {
    auto buf_len = dname.size() + 64;
    char* buf = new char[buf_len];
    int n = 0;
    snprintf(buf, buf_len, "%s-%05d", dname.c_str(), n);

    string ret;
    while(++n < max_tries) {
      if(try_make_dir(string(buf))) {
        ret = string(buf);
        break;
      }
    }

    delete[] buf;
    return ret;
  }
}

}; // fs


}; // utils


