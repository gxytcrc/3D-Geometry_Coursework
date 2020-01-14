#ifndef MYTOOLS
#define MYTOOLS

#define NOMINMAX

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <igl/vertex_triangle_adjacency.h>
#include <nanoflann.hpp>
#include <igl/fit_plane.h>
#include <igl/adjacency_list.h>
#include<igl/adjacency_matrix.h>
#include <Eigen/Sparse>
#include<Eigen/core>
#include<Eigen/Sparsecore>

void get_cube(Eigen::MatrixXd & out_V , Eigen::MatrixXi & out_F);


void calculate_vertex_normal(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	Eigen::MatrixXd const & FN,
	Eigen::MatrixXd & out_VN);

void calculate_vertex_normal_flann(
	Eigen::MatrixXd const & V, 
	Eigen::MatrixXi const & F,  
	Eigen::MatrixXd & out_VN);

void Rotation_XC(double & degree_X, Eigen::Matrix3d & out_rotation);
void Rotation_YC(double & degree_Y, Eigen::Matrix3d & out_rotation);
void Rotation_ZC(double & degree_Z, Eigen::Matrix3d & out_rotation);

void comput_uniform(Eigen::MatrixXd  & m_V, Eigen::MatrixXi & m_F, Eigen::MatrixXd & FN, std::vector<std::vector<double> >& A,
	std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::MatrixXd & H);
void compute_M(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::MatrixXd & Area, Eigen::SparseMatrix<double> & M);
void C_Matrix_fomulation(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::SparseMatrix<double> & C_Matrix);
void compute_Gaussian(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, Eigen::MatrixXd & Area, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::MatrixXd & Gaussian_C);
void Convert_M(Eigen::SparseMatrix<double> & M, Eigen::SparseMatrix<double> & M_half, Eigen::SparseMatrix<double> & M_Positive);
void Convert_C(Eigen::SparseMatrix<double> & C_Matrix, Eigen::SparseMatrix<double> & C_Matrix_ab);
void Formate_Eigen_V(Eigen::MatrixXd & m_V, Eigen::SparseMatrix<double> & M_Positive, Eigen::MatrixXd & e_vectors, Eigen::MatrixXd & new_V);
#endif


