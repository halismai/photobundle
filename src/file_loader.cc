#include "file_loader.h"
#include "utils.h"
#include "debug.h"

FileLoader::FileLoader(std::string dname, std::string pattern, int frame_start)
    : _frame_start(frame_start), _frame_counter(0)
{
  auto d = utils::fs::expand_tilde(dname + utils::fs::dirsep(dname));

  {
    auto err_msg = utils::Format("directory '%s' does not exist", d.c_str());
    THROW_ERROR_IF( !utils::fs::exists(d), err_msg.c_str());
  }

  if(pattern.find('%') == std::string::npos)
  {
    auto glob_pattern = d + pattern;
    _files = utils::glob(glob_pattern);

    if((int) _files.size() > _frame_start) {
      _files.erase(_files.begin(), _files.begin() + _frame_start);
    } else {
      Warn("frame start exceeds the number of frames [%d/%zu]\n",
           _frame_start, _files.size());
    }
  } else
  {
    // delay reading all the filename for large dataests and store only the
    // printf style format needed to load a number frame
    _fmt = d + pattern;
  }
}

std::string FileLoader::operator[](size_t i) const
{
  if(!_files.empty()) {
    return i < _files.size() ? _files[i] : "";
  } else {
    return utils::Format(_fmt.c_str(), _frame_start + (int) i);
  }
}

size_t FileLoader::size() const { return _files.size(); }

std::string FileLoader::next()
{
  return this->operator[](_frame_counter++);
}

