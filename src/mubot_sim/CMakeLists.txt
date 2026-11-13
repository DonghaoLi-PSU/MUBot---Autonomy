cmake_minimum_required(VERSION 3.8)
project(mubot_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)   
find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED )
find_package(gz-msgs10 REQUIRED)

# ros2 launch file
install(DIRECTORY launch meshes urdf worlds
  DESTINATION share/${PROJECT_NAME})

# Hydrodynamics system plugin
foreach(plugin actuator hydro)
  add_library(mubot_${plugin} SHARED src/${plugin}_plugin.cc)
  target_include_directories(mubot_${plugin} PRIVATE include)
  target_compile_features(mubot_${plugin} PRIVATE cxx_std_17)
  target_link_libraries(mubot_${plugin} PRIVATE
    gz-sim8::gz-sim8 
    gz-plugin2::gz-plugin2
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)
  if(ament_index_cpp_FOUND)
    target_link_libraries(mubot_${plugin} PRIVATE ament_index_cpp::ament_index_cpp)
  endif()
  install(TARGETS mubot_${plugin} LIBRARY DESTINATION lib/${PROJECT_NAME})
endforeach()

install(PROGRAMS
  scripts/EM_pgpe.py
  scripts/gazebo_connection.py
  scripts/mubot_rob_env.py
  scripts/start_empgpe.py
  DESTINATION lib/${PROJECT_NAME}
  )


ament_package()