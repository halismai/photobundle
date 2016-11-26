file(GLOB src "*.cc")
foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})
  target_link_libraries(${bname} photobundle ${MY_LIBRARIES})
endforeach()

