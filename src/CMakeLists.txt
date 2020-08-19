set(SOURCES
	image.cpp
	stitcher_impl.cpp
	stitcher.cpp
	math_util.cpp
	profile_timer.cpp
	project_cam.cu
	blend.cu
	)

add_library(gpustitch ${SOURCES})
add_library(gpustitch::gpustitch ALIAS gpustitch)
target_compile_features(gpustitch PUBLIC cxx_std_11)
set_target_properties(gpustitch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpustitch PRIVATE cudart)
target_include_directories(gpustitch
	PUBLIC
	$<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	)

add_executable(stitchtest main.cpp)
target_compile_features(stitchtest PRIVATE cxx_std_17)
target_include_directories(stitchtest
	#PUBLIC
	#$<INSTALL_INTERFACE:include>
	#$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	)
find_package(PNG REQUIRED)
target_link_libraries(stitchtest PRIVATE
	gpustitch::gpustitch
	PNG::PNG
	cudart)
