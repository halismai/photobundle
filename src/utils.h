#ifndef UTILS_GLOB_H
#define UTILS_GLOB_H

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <map>

#include <boost/program_options.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/log/trivial.hpp>

#include "debug.h"

namespace std {
inline std::string to_string(std::string s) { return s; }
}; // std



namespace utils {

template <typename T> inline void UNUSED(T&&) {}

/** case insenstive string comparision */
bool icompare(const std::string& a, const std::string& b);

struct CaseInsenstiveComparator
{
  bool operator()(const std::string&, const std::string&) const;
}; // CaseInsenstiveComparator

// based on
// http://stackoverflow.com/questions/17835689/accessing-boost-fusion-map-field-name
template <class Sequence>
struct PrintFields
{
  PrintFields(const Sequence& seq, std::ostream& os,
              std::string delim = " = ")
      : _seq(seq), _os(os), _delim(delim) {}

  template <typename Index> inline
  void operator()(Index idx) const
  {
    using namespace boost::fusion;

    std::string fname = extension::struct_member_name<Sequence,idx>::call();
    _os << fname << " = " << at<Index>(_seq) << "\n";
  }

 private:
  const Sequence& _seq;
  std::ostream& _os;
  std::string _delim;
};

template <class Sequence> inline
std::ostream& toStream(const Sequence& seq, std::ostream& os = std::cout)
{
  using namespace boost::mpl;
  using namespace boost::fusion;

  typedef range_c<std::size_t, 0, result_of::size<Sequence>::value> Indices;
  for_each(Indices(), PrintFields<Sequence>(seq, os));

  os << "\b" << "";
  return os;
}

template <class T>
typename std::enable_if<std::is_integral<T>::value, std::uniform_int_distribution<T>>::type
getUniformDistribution(T max_val)
{
  return std::uniform_int_distribution<T>(0, max_val);
}

template <class T>
typename std::enable_if<std::is_floating_point<T>::value, std::uniform_real_distribution<T>>::type
getUniformDistribution(T max_val)
{
  return std::uniform_real_distribution<T>(0, max_val);
}

template <typename T>
std::vector<T> generateRandomVector(size_t N, T max_val = std::is_integral<T>::value ? T(255) : T(1))
{
  std::mt19937 rng;
  auto dist = getUniformDistribution<T>(max_val);
  auto gen = std::bind(dist, rng);

  std::vector<T> ret(N);
  std::generate(std::begin(ret), std::end(ret), gen);
  return ret;
}

/** based on Facebook's Folly code */
template <class T>
void doNotOptimizeAway(T&& d)
{
  asm volatile("" : "+r"(d));
}

/**
 * converts the input string to a number
 */
template <typename T> T str2num(const std::string&);

template <> int str2num<int>(const std::string& s);
template <> double str2num<double>(const std::string& s);
template <> float str2num<float>(const std::string& s);
template <> bool str2num<bool>(const std::string& s);

/**
 * converts string to number
 * \return false if conversion to the specified type 'T' fails
 *
 * e.g.:
 *
 * double num;
 * assert( true == str2num("1.6", num) );
 * assert( false == str2num("hello", num) );
 *
 */
template <typename T> inline
bool str2num(std::string str, T& num)
{
  std::istringstream ss(str);
  return !(ss >> num).bad();
}

/**
 * Uses the delimiter to split the string into tokens of numbers, e.g,
 *
 * string str = "1.2 1.3 1.4 1.5";
 * auto tokens = splitstr(str, ' ');
 *
 * // 'tokens' now has [1.2, 1.3, 1.4, 1.5]
 */
std::vector<std::string> splitstr(const std::string& str, char delim = ' ');

template <typename T> inline
std::vector<T> str2num(const std::vector<std::string>& strs)
{
  std::vector<T> ret(strs.size());
  for(size_t i = 0; i < strs.size(); ++i)
    ret[i] = str2num<T>(strs[i]);

  return ret;
}


template <typename T> inline
std::string tostring(const T& something)
{
  std::stringstream oss;
  oss << something;

  return oss.str();
}


/**
 * globs for files or directories matching the pattern
 * \return the matched files, or an empty vector on any error
 *
 * Example
 *
 *   using namespace std;
 *   vector<string> files = glob("~/data/image*.png");
 *   for(const auto& f : files)
 *      system(("identify " + f).c_str());
 *
 */
std::vector<std::string> glob(const std::string& pattern, bool verbose=true);


/**
 * ProgramOptions
 */
class ProgramOptions
{
 public:
  ProgramOptions(std::string name = "ProgramOptions");

  ProgramOptions& addOption(std::string name, std::string msg);

  inline ProgramOptions& operator()(std::string name, std::string msg) {
    return addOption(name, msg);
  }

  template <class T> inline
  ProgramOptions& operator()(std::string name, T v, std::string help)
  {
    _desc.add_options()(name.c_str(), boost::program_options::value<T>()->default_value(v),
                        help.c_str());
    return *this;
  }

  ProgramOptions& operator()(std::string name, const char* v, std::string help)
  {
    return this->operator()(name, std::string(v), help);
  }

  template <class T> inline
  T get(std::string name) const {
    try {
      return _vm[name].template as<T>();
    } catch(const std::exception& ex) {
      std::cerr << "Error: " << ex.what() << std::endl;
      throw ex;
    }
  }

  void parse(int argc, char** argv);

  void printHelpAndExit(int exit_code = 0) const;

  bool hasOption(std::string) const;

 private:
  boost::program_options::options_description _desc;
  boost::program_options::variables_map _vm;
}; // ProgramOptions

/** vsprintf like */
std::string Format(const char* fmt, ...);

struct Error : public std::logic_error
{
  inline Error(std::string what)
      : logic_error(what) {}
}; // Error


#define THROW_ERROR(msg) \
    throw utils::Error(utils::Format("[ %s:%04d ] %s", MYFILE, __LINE__, msg))

#define THROW_ERROR_IF(cond, msg) if( !!(cond) ) THROW_ERROR( (msg) )

#define DIE_IF(cond, msg) if( !!(cond) ) Fatal((msg))

/**
 * simple config file with data of the form
 *   VarName = Value
 *
 * Lines that begin with a '#' or '%' are treated as comments
 *
 * Example usage
 *
 *    ConfigFile cf("myfile.cfg");
 *
 *    // get a variable with a default value if it does not exist in myfile.cfg
 *    auto v = cf.get<std::string>("MyVariable", "default");
 *
 *    // this will throw an error if the variable does not exist
 *    try {
 *      auto required_var = cf.get<int>("VariableName");
 *    } catch(const std::exception& ex) {
 *      std::cerr << "value 'VariableName' is required\n";
 *    }
 *
 *    // We can also print the contents of the ConfigFile
 *    std::cout << cf << std::endl;
 *
 *    // or add new values
 *    cf.set<double>("MyNewVariable", 1.618);
 *
 *    // and write it to disk
 *    cf.save("newfile.cfg");
 *
 */
class ConfigFile
{
 public:
  /**
   * default constructor, does not do anything
   */
  ConfigFile();

  /**
   * Loads a config file from 'filename'.
   *
   * \throw Error if filename does not exist
   */
  ConfigFile(std::string filename);

  /**
   * Loads the config from an opened ifstream
   * \throw Error if 'ifs' is not open
   */
  ConfigFile(std::ifstream& ifs);

  /**
   * Writes the contents of the file to disk
   *
   * \param filename output filename
   * \return true if successfull
   */
  bool save(std::string filename) const;

  /**
   * Get the value named 'var_name'
   *
   * \throw Error if 'var_name' does not exist, or conversion to the required
   * type 'T' fails
   */
  template <typename T> inline
  T get(std::string var_name) const;

  /**
   * Get the value name 'var_name'
   *
   * If any error occurs (e.g. var_name does not exist) the function will
   * silently return the supplied 'default_val'
   */
  template <typename T> inline
  T get(std::string var_name, const T& default_val) const;

  /**
   * Sets 'var_name' to the specified value
   */
  template <typename T> inline
  ConfigFile& set(std::string var_name, const T& value);

  /**
   * sets values with method chaining. For example,
   *
   * ConfigFile cf;
   * cf("SpeedOfLight", "299792458.0")
   *   ("PI",           "3.14159265359")
   *   ("PHI",          "1.618033988749895").save("my_awesome_config.cfg");
   */
  ConfigFile& operator()(const std::string&, const std::string&);

  friend std::ostream& operator<<(std::ostream&, const ConfigFile&);

 protected:
  void parse(std::ifstream&);

  std::map<std::string, std::string, CaseInsenstiveComparator> _data;
}; // ConfigFile


template <typename T>
T ConfigFile::get(std::string name) const
{
  const auto& value_it = _data.find(name);
  if(value_it == _data.end())
    throw Error("no key " + name);

  T ret;
  if(!str2num(value_it->second, ret))
    throw Error("failed to convert '" + value_it->second +
                "' to type " + typeid(T).name());

  return ret;
}

template <typename T>
T ConfigFile::get(std::string name, const T& default_val) const
{
  try {
    return get<T>(name);
  } catch(const std::exception& ex) {
    /*Warn("ConfigFile: get %s Error: %s [using default %s]\n",
         name.c_str(), ex.what(), std::to_string(default_val).c_str()); */
    return default_val;
  }
}

template <typename T> inline
ConfigFile& ConfigFile::set(std::string name, const T& value)
{
  _data[name] = std::to_string(value);
  return *this;
}

/**
 * sleep for the given milliseconds
 */
void Sleep(int32_t milliseconds);


template <typename Iterator> static inline typename
Iterator::value_type median(Iterator first, Iterator last)
{
  auto n = std::distance(first, last);
  auto middle = first + n/2;
  std::nth_element(first, middle, last);
  //__gnu_parallel::nth_element(first, middle, last);

  if(n % 2 != 0) {
    return *middle;
  } else {
    auto m = std::max_element(first, middle);
    return (*m + *middle) / 2.0;
  }
}

template <class Container> static inline typename Container::
value_type median(Container& data)
{
  if(data.empty()) {
    BOOST_LOG_TRIVIAL(warning) << "median() : empty vector";
    return typename Container::value_type(0);
  }

  if(data.size() < 3)
    return data[0];

  return median(std::begin(data), std::end(data));
}

namespace fs {

/**
 * \return directory separator, this is a slash '/'
 */
std::string dirsep(std::string fn);

/**
 * Expands '~' to user's home directory
 */
std::string expand_tilde(std::string);


/**
 * \return the extension of the input filename
 */
std::string extension(std::string filename);

/**
 * \return true if path exists
 */
bool exists(std::string path);

/**
 * \return true if path is a regular file
 */
bool is_regular(std::string path);

/**
 * \return true if directory
 */
bool is_dir(std::string path);

/**
 * Creates a directory.
 *
 * \return name of the directory that was created (empty if we could not create
 * one for you)
 *
 * if 'try_unique' is true, the function will keep trying up to 'max_tries' to
 * create a unique directory
 */
std::string mkdir(std::string dname, bool try_unique = false, int max_tries = 1000);


}; // fs


}; // utils

#endif // GLOB_H

