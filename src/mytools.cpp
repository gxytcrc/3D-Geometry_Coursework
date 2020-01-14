#include "mytools.h"
#include<set>
#include<vector>
#include<set>
#include<vector>
#include<iostream>
#include<list>
#include<cmath>
#include<igl/readOBJ.h>
#include<igl/vertex_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>


void get_cube(Eigen::MatrixXd & out_V, Eigen::MatrixXi & out_F)
{
	out_V = (Eigen::MatrixXd(8, 3) <<
		0.0, 0.0, 0.0,
		0.0, 0.0, 1.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 1.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		1.0, 1.0, 0.0,
		1.0, 1.0, 1.0).finished();

	out_F = (Eigen::MatrixXi(12, 3) <<
		1, 7, 5,
		1, 3, 7,
		1, 4, 3,
		1, 2, 4,
		3, 8, 7,
		3, 4, 8,
		5, 7, 8,
		5, 8, 6,
		1, 5, 6,
		1, 6, 2,
		2, 6, 8,
		2, 8, 4).finished().array() - 1;
}


void calculate_vertex_normal(Eigen::MatrixXd const & V, Eigen::MatrixXi const & F, Eigen::MatrixXd const & FN, Eigen::MatrixXd & out_VN)
{
	//
	// input:
	//   V: vertices
	//   F: face 
	//   FN: face normals
	// output:
	//   out_VN
	//
	//   Your job is to implement vertex normal calculation
	//
	out_VN.resize(V.rows(), V.cols());
	out_VN.setZero();
	
	for (int i = 0; i < V.rows(); i++) {
		std::set<int> s;
		std::set<int>::iterator m;

		for (int row = 0; row < F.rows(); row++) {
			for (int col = 0; col < 3; col++) {
				if (F(row, col) == i) {
					s.insert(row);
				}
			}
		}
		if (s.size() != 0) {
			Eigen::MatrixXd sum(1, 3);
			sum.setZero();
			for (m = s.begin(); m != s.end(); m++) {
				for (int n = 0; n < 3; n++) {
					sum(0, n) = sum(0, n) + FN(*m, n);
				}
			}
			for (int n = 0; n < 3; n++) {
				sum(0, n) = sum(0, n) / s.size();
			}
			out_VN.row(i) = sum.row(0);
		}
	}


}

void calculate_vertex_normal_flann(Eigen::MatrixXd const & V, Eigen::MatrixXi const & F, Eigen::MatrixXd & out_VN)
{
	//
	// input:
	//   V: vertices
	//   F: face 
	//   FN: face normals
	// output:
	//   out_VN
	//
	// Your job is to implement vertex normal calculation vis using flann and igl:fitplane
	//  
	// igl::fit_plane(V, N, C);
	// Input:
	//   V #Vx3 matrix. The 3D point cloud, one row for each vertex.
	// Output: 
	//   N 1x3 Vector. The normal of the fitted plane.
	//   C 1x3 Vector. A point that lies in the fitted plane.
	//

	out_VN.resize(V.rows(), V.cols());
	out_VN.setZero();



}

void Rotation_XC(double & degree_X, Eigen::Matrix3d & out_rotation)
{
	out_rotation.setZero();
	out_rotation(1, 1) = cos(degree_X); out_rotation(1, 2) = -sin(degree_X);
	out_rotation(2, 1) = sin(degree_X); out_rotation(2, 2) = cos(degree_X);
	out_rotation(0, 0) = 1;
}
void Rotation_YC(double & degree_Y, Eigen::Matrix3d & out_rotation) 
{
	out_rotation.setZero();
	out_rotation(0, 0) = cos(degree_Y); out_rotation(0, 2) = -sin(degree_Y);
	out_rotation(2, 0) = sin(degree_Y); out_rotation(2, 2) = cos(degree_Y);
	out_rotation(1, 1) = 1;
}

void Rotation_ZC(double & degree_Z, Eigen::Matrix3d & out_rotation)
{
	out_rotation.setZero();
	out_rotation(0, 0) = cos(degree_Z); out_rotation(0, 1) = -sin(degree_Z);
	out_rotation(1, 0) = sin(degree_Z); out_rotation(1, 1) = cos(degree_Z);
	out_rotation(2, 2) = 1;
}

void comput_uniform(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, Eigen::MatrixXd & FN, std::vector<std::vector<double> >& A, 
	std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V,Eigen::MatrixXd & H)
{
	using namespace std;
	for (int row = 0; row < m_V.rows(); row++) {

		//obtain the neibor point numer
		int Size_A = A[row].size();

		//first part is calculating the mean H, first step is calculating the delta X
		Eigen::Vector3d det_X(0,0,0);
		Eigen::Vector3d normal_V;
		for (int i = 0; i < Size_A; i++) {
			for (int j = 0; j < 3; j++) {
				det_X[j] = det_X[j] + (m_V(A[row][i], j) - m_V(row, j));
			}
		}
		normal_V = FN.row(row);
		det_X = det_X / Size_A;
		H(row, 0) = (det_X.norm() / normal_V.norm())/2;
	}
}
void compute_M(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::MatrixXd & Area, Eigen::SparseMatrix<double> & M)
{
	using namespace std;
	Area.resize(m_V.rows(), 1);
	M.resize(m_V.rows(), m_V.rows());
	for (int row = 0; row < m_V.rows(); row++) {
		Eigen::Vector3d edge_1;
		Eigen::Vector3d edge_2;
		Eigen::Vector3d face1;
		double sub_angle;
		double sub_area;
		double sum_area = 0;
		double product;
		for (int i = 0; i < Nei_F[row].size(); i++) {
			int index_F = Nei_F[row][i];
			for (int j = 0; j < 3; j++) {
				face1[j] = m_F(index_F, j);
			}
			if (F_V[row][i] == 0) {
				edge_1 = m_V.row(face1[1]) - m_V.row(row);
				edge_2 = m_V.row(face1[2]) - m_V.row(row);

			}
			else if (F_V[row][i] == 1) {
				edge_1 = m_V.row(face1[0]) - m_V.row(row);
				edge_2 = m_V.row(face1[2]) - m_V.row(row);
			}
			else if (F_V[row][i] == 2) {
				edge_1 = m_V.row(face1[0]) - m_V.row(row);
				edge_2 = m_V.row(face1[1]) - m_V.row(row);
			}
			product = (edge_1.dot(edge_2)) / (edge_1.norm() * edge_2.norm());
			sub_angle = acos(product);
			sub_area = 0.5 * edge_1.norm() * edge_2.norm() * sin(sub_angle);
			sum_area = sum_area + sub_area;
		}
		sum_area = sum_area / 3;
		Area(row, 0) = sum_area;
		M.insert(row, row) = 0.5 / sum_area;
	}
}
void C_Matrix_fomulation(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::SparseMatrix<double> & C_Matrix)
{
	using namespace std;
	C_Matrix.resize(m_V.rows(), m_V.rows());
	for (int row = 0; row < m_V.rows(); row++) {
		Eigen::Vector3d edge_1;
		Eigen::Vector3d edge_2;
		Eigen::Vector3d edge_3;
		Eigen::Vector3d edge_4;
		Eigen::Vector3d face1;
		Eigen::MatrixXd Dense = Eigen::MatrixXd::Zero(m_V.rows(), 1);
		double angle_1;
		double angle_2;
		double sum_D;

		for (int i = 0; i < Nei_F[row].size(); i++) {
			int index_F = Nei_F[row][i];
			for (int j = 0; j < 3; j++) {
				face1[j] = m_F(index_F, j);
			}

			if (F_V[row][i] == 0) {
				edge_1 = m_V.row(face1[2]) - m_V.row(face1[1]);
				edge_2 = m_V.row(face1[1]) - m_V.row(face1[2]);
				edge_3 = m_V.row(row) - m_V.row(face1[1]);
				edge_4 = m_V.row(row) - m_V.row(face1[2]);
				angle_1 = acos((edge_1.dot(edge_3)) / (edge_1.norm() * edge_3.norm()));
				angle_2 = acos((edge_2.dot(edge_4)) / (edge_2.norm() * edge_4.norm()));
				//C_Matrix(row, face1[2]) = C_Matrix(row, face1[2]) + 1 / tan(angle_1);
				//C_Matrix(row, face1[1]) = C_Matrix(row, face1[1]) + 1 / tan(angle_2);
				//C_Matrix(row, row) = C_Matrix(row, row) - 1 / tan(angle_1) - 1 / tan(angle_2);
				Dense(face1[2], 0) = Dense(face1[2], 0) + 1 / tan(angle_1);
				Dense(face1[1], 0) = Dense(face1[1], 0) + 1 / tan(angle_2);
			}
			else if (F_V[row][i] == 1) {
				edge_1 = m_V.row(face1[2]) - m_V.row(face1[0]);
				edge_2 = m_V.row(face1[0]) - m_V.row(face1[2]);
				edge_3 = m_V.row(row) - m_V.row(face1[0]);
				edge_4 = m_V.row(row) - m_V.row(face1[2]);
				angle_1 = acos((edge_1.dot(edge_3)) / (edge_1.norm() * edge_3.norm()));
				angle_2 = acos((edge_2.dot(edge_4)) / (edge_2.norm() * edge_4.norm()));
				//C_Matrix(row, face1[2]) = C_Matrix(row, face1[2]) + 1 / tan(angle_1);
				//C_Matrix(row, face1[0]) = C_Matrix(row, face1[0]) + 1 / tan(angle_2);
				//C_Matrix(row, row) = C_Matrix(row, row) - 1 / tan(angle_1) - 1 / tan(angle_2);
				Dense(face1[2], 0) = Dense(face1[2], 0) + 1 / tan(angle_1);
				Dense(face1[0], 0) = Dense(face1[0], 0) + 1 / tan(angle_2);

			}
			else if (F_V[row][i] == 2) {
				edge_1 = m_V.row(face1[1]) - m_V.row(face1[0]);
				edge_2 = m_V.row(face1[0]) - m_V.row(face1[1]);
				edge_3 = m_V.row(row) - m_V.row(face1[0]);
				edge_4 = m_V.row(row) - m_V.row(face1[1]);
				angle_1 = acos((edge_1.dot(edge_3)) / (edge_1.norm() * edge_3.norm()));
				angle_2 = acos((edge_2.dot(edge_4)) / (edge_2.norm() * edge_4.norm()));
				//C_Matrix(row, face1[1]) = C_Matrix(row, face1[1]) + 1 / tan(angle_1);
				//C_Matrix(row, face1[0]) = C_Matrix(row, face1[0]) + 1 / tan(angle_2);
				//C_Matrix(row, row) = C_Matrix(row, row) - 1 / tan(angle_1) - 1 / tan(angle_2);
				Dense(face1[1], 0) = Dense(face1[1], 0) + 1 / tan(angle_1);
				Dense(face1[0], 0) = Dense(face1[0], 0) + 1 / tan(angle_2);
			}
		
		}
		sum_D = Dense.sum();
		C_Matrix.insert(row, row) = - sum_D;
		for (int i = 0; i < m_V.rows(); i++) {
			if (Dense(i, 0) != 0) {
				C_Matrix.insert(row, i) = Dense(i, 0);
			}
		}
	}
}
void compute_Gaussian(Eigen::MatrixXd & m_V, Eigen::MatrixXi & m_F, Eigen::MatrixXd & Area, std::vector<std::vector<double> >& Nei_F, std::vector<std::vector<double> >& F_V, Eigen::MatrixXd & Gaussian_C)
{
	using namespace std;
	Gaussian_C.resize(m_V.rows(), 1);
	for (int row = 0; row < m_V.rows(); row++) {
		Eigen::Vector3d edge_1;
		Eigen::Vector3d edge_2;
		Eigen::Vector3d face1;
		double sub_angle;
		double sub_area;
		double sum_area = 0;
		double product;
		double angle = 0;
		for (int i = 0; i < Nei_F[row].size(); i++) {
			int index_F = Nei_F[row][i];
			for (int j = 0; j < 3; j++) {
				face1[j] = m_F(index_F, j);
			}
			if (F_V[row][i] == 0) {
				edge_1 = m_V.row(face1[1]) - m_V.row(row);
				edge_2 = m_V.row(face1[2]) - m_V.row(row);

			}
			else if (F_V[row][i] == 1) {
				edge_1 = m_V.row(face1[0]) - m_V.row(row);
				edge_2 = m_V.row(face1[2]) - m_V.row(row);
			}
			else if (F_V[row][i] == 2) {
				edge_1 = m_V.row(face1[0]) - m_V.row(row);
				edge_2 = m_V.row(face1[1]) - m_V.row(row);
			}
			product = (edge_1.dot(edge_2)) / (edge_1.norm() * edge_2.norm());
			sub_angle = acos(product);
			angle = angle + sub_angle;
			if (row == 0) {
				cout << sub_angle << " ";
			}
		}
		Gaussian_C(row, 0) = (2 * 3.1415926 - angle) / Area(row, 0);

	}
}
void Convert_M(Eigen::SparseMatrix<double> & M, Eigen::SparseMatrix<double> & M_half, Eigen::SparseMatrix<double> & M_Positive)
{
	M_half.resize(M.rows(), M.cols());
	M_Positive.resize(M.rows(), M.cols());
	Eigen::MatrixXd M_Dense(M.rows(), M.cols());
	M_Dense = M.toDense();
	for (int i = 0; i < M.rows(); i++) {
		M_half.insert(i, i) = sqrt(M_Dense(i, i));
		M_Positive.insert(i, i) = (1 / M_Dense(i, i));
	}
}
void Convert_C(Eigen::SparseMatrix<double> & C_Matrix, Eigen::SparseMatrix<double> & C_Matrix_ab) 
{
	C_Matrix_ab.resize(C_Matrix.rows(), C_Matrix.cols());
	Eigen::MatrixXd C_Dense(C_Matrix.rows(), C_Matrix.cols());
	C_Dense = C_Matrix.toDense();
	for (int i = 0; i < C_Dense.rows(); i++) {
		for (int j = 0; j < C_Dense.cols(); j++) {
			if (C_Dense(i, j) != 0) {
				C_Matrix_ab.insert(i, j) = abs(C_Dense(i, j));
			}
		}
	}
}
void Formate_Eigen_V(Eigen::MatrixXd & m_V, Eigen::SparseMatrix<double> & M_Positive, Eigen::MatrixXd & e_vectors, Eigen::MatrixXd & new_V)
{
	using namespace std;
	using namespace Eigen;

	new_V.resize(m_V.rows(), m_V.cols());

	Eigen::MatrixXd EEvector(e_vectors.rows(), 1);
	Eigen::MatrixXd X_Vector(m_V.rows(), 1);
	Eigen::MatrixXd Y_Vector(m_V.rows(), 1);
	Eigen::MatrixXd Z_Vector(m_V.rows(), 1);
	Eigen::MatrixXd X_Matrix(m_V.rows(), 1);
	Eigen::MatrixXd Y_Matrix(m_V.rows(), 1);
	Eigen::MatrixXd Z_Matrix(m_V.rows(), 1);
	Eigen::MatrixXd X_Matrix_m(m_V.rows(), 1);
	Eigen::MatrixXd Y_Matrix_m(m_V.rows(), 1);
	Eigen::MatrixXd Z_Matrix_m(m_V.rows(), 1);
	double X_value;
	double Y_value;
	double Z_value;


	X_Matrix.setZero();
	Y_Matrix.setZero();
	Z_Matrix.setZero();
	X_Vector = m_V.col(0);
	Y_Vector = m_V.col(1);
	Z_Vector = m_V.col(2);

	for (int k = 0; k < e_vectors.cols(); k++) {
		EEvector = e_vectors.col(k);
		X_Matrix_m = (X_Vector.transpose() * M_Positive * EEvector);
		Y_Matrix_m = (Y_Vector.transpose() * M_Positive * EEvector);
		Z_Matrix_m = (Z_Vector.transpose() * M_Positive * EEvector);
		X_value = X_Matrix_m(0, 0);
		Y_value = Y_Matrix_m(0, 0);
		Z_value = Z_Matrix_m(0, 0);
		X_Matrix = X_Matrix + X_value * EEvector;
		Y_Matrix = Y_Matrix + Y_value * EEvector;
		Z_Matrix = Z_Matrix + Z_value * EEvector;
	}


	new_V.col(0) = X_Matrix;
	new_V.col(1) = Y_Matrix;
	new_V.col(2) = Z_Matrix;
}