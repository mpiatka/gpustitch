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
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(gpustitch PRIVATE ${CUDART_LIBRARY})
target_include_directories(gpustitch
	PUBLIC
	$<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)

add_executable(stitchtest main.cpp)
target_compile_features(stitchtest PRIVATE cxx_std_17)
target_include_directories(stitchtest
	#PUBLIC
	#$<INSTALL_INTERFACE:include>
	#$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)
find_package(PNG REQUIRED)
target_link_libraries(stitchtest PRIVATE
	gpustitch::gpustitch
	PNG::PNG
	${CUDART_LIBRARY})

install(TARGETS gpustitch
	EXPORT gpustitch-export
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include/libgpustitch
	RUNTIME DESTINATION bin)

install(EXPORT gpustitch-export
	FILE gpustitchConfig.cmake
	NAMESPACE gpustitch::
	DESTINATION /lib/cmake/gpustitch)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include
	DESTINATION include/libgpustitch
	FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

configure_file(
	${PROJECT_SOURCE_DIR}/src/libgpustitch.pc.in
	${PROJECT_BINARY_DIR}/src/libgpustitch.pc
	@ONLY
	)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgpustitch.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
