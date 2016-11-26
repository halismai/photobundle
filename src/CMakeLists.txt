file(GLOB src "*.cc")
add_library(photobundle ${LIBRARY_TYPE} ${src})
set_target_properties(photobundle PROPERTIES LINKER_LANGUAGE CXX)

if(WITH_OPENCV)
  target_link_libraries(photobundle ${OpenCV_LIBS})
endif()



