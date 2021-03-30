cmake_minimum_required(VERSION 3.17)
project(Exercicios_222_a_226)

set(CMAKE_CXX_STANDARD 14)

add_executable(Exercicios_222_a_226 E222_a_226.cpp)

#Acesso ao GSL-GNU
target_link_libraries(Exercicios_222_a_226 -lgsl -lgslcblas -lm)

#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES E222_a_226.cpp)