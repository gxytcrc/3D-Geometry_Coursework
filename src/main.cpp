#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <igl/file_exists.h>
#include <imgui/imgui.h>
#include <iostream>
#include <random>
#include<igl/readPLY.h>
#include <Eigen/SVD> 
#include"mytools.h"
#include<set>
#include<vector>
#include<iostream>
#include<list>
#include<cmath>
#include<igl/readOBJ.h>
#include<igl/vertex_triangle_adjacency.h>
#include "igl/parula.h"
#include "igl/colormap.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/GenEigsComplexShiftSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>







class MyContext
{
public:

	MyContext() :Display(0), mode(0), iteration(1)
	{
	
		//insert the ply point 
		igl::readOBJ("../data/example_meshes/cow.obj", m_V, m_F);
		igl::readOBJ("../data/example_meshes/bunny.obj", m_V0, m_F0);
		//initial vertex input of question 5
		m_V5 = m_V0;
		m_V6 = m_V0;

		m_V70 = m_V0;
		const double mean = 0.0;//mean
		const double stddev = 0.001;//standard divation
		std::default_random_engine generator;
		std::normal_distribution<double> dist(mean, stddev);
		for (int i = 0; i < m_V70.rows(); i++) {
			for (int j = 0; j < 3; j++) {
				m_V70(i, j) = m_V70(i, j) + dist(generator);
			}
		}
		m_V7 = m_V70;

	}
	~MyContext() {}

	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;
	Eigen::MatrixXd m_V0;
	Eigen::MatrixXi m_F0;
	Eigen::MatrixXd m_FN;
	Eigen::MatrixXd m_VN;
	Eigen::MatrixXd m_VN_flann;
	Eigen::MatrixXd m_G;
	Eigen::MatrixXd m_H;
	Eigen::MatrixXd m_C;
	Eigen::MatrixXd m_CG;
	//set the vertex value of question 5
	Eigen::MatrixXd m_V5;
	Eigen::MatrixXd m_V6;
	Eigen::MatrixXd m_V7;
	Eigen::MatrixXd m_V70;


	int Display;
	int mode;
	int iteration;
	double iter;

	void reset_display(igl::opengl::glfw::Viewer& viewer)
	{

		static std::default_random_engine generator;
		
		viewer.data().clear();

		if (mode == 0)
		{
			using namespace std;
			Eigen::MatrixXd area;
			Eigen::SparseMatrix<double> M;
			Eigen::MatrixXd FN;
			Eigen::MatrixXd H(m_V.rows(),1);
			Eigen::MatrixXd Gaussian_C(m_V.rows(), 1);
			Eigen::MatrixXd Gaussian_B(m_V.rows(), 1);
			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			std::vector<vector<double> > A;
			//obtain the vertex normal
			igl::per_vertex_normals(m_V, m_F, FN);
			//obtain the neibor vertexes of every vertex
			igl::adjacency_list(m_F, A);
			//obtain the neibor face
			igl::vertex_triangle_adjacency(m_V.rows(),m_F,Nei_F,F_V);
			//compute the question 1
			comput_uniform(m_V, m_F, FN, A, Nei_F, F_V, H);
			compute_M(m_V, m_F, Nei_F, F_V, area, M);
			compute_Gaussian(m_V, m_F, area, Nei_F, F_V, Gaussian_C);

			//cout << Gaussian_C;
			cout << A[0].size();
			
			//cout << H;
			m_H = 3*(H.array() - H.minCoeff()) / (H.maxCoeff() - H.minCoeff());
			igl::parula(m_H, false, m_C);
			m_G = 30 * (Gaussian_C.array()) / (Gaussian_C.maxCoeff() - Gaussian_C.minCoeff());
			igl::parula(m_G, false, m_CG);
			std::cout << "eigen version.:" << EIGEN_WORLD_VERSION << "," << EIGEN_MAJOR_VERSION << EIGEN_MINOR_VERSION << "\n";
			//viewer.data().add_points(m_V.row(0), Eigen::RowVector3d(1, 0, 0));
			//viewer.data().add_points(m_V, Eigen::RowVector3d(0, 0, 0));
			if (Display == 1) {

				viewer.data().set_mesh(m_V, m_F);
				viewer.data().set_colors(m_C);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;
			}

			else if (Display == 2) {

				viewer.data().set_mesh(m_V, m_F);
				viewer.data().set_colors(m_CG);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;
			
			}

			else if (Display == 0) {
				viewer.data().set_mesh(m_V, m_F);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;
			}
		}
		else if(mode ==1)
		{	
			using namespace std;
			
			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			Eigen::MatrixXd area;
			Eigen::SparseMatrix<double> C_matrix;
			Eigen::SparseMatrix<double> M;
			Eigen::MatrixXd HN;
			Eigen::MatrixXd H;
			Eigen::MatrixXd Gaussian_C;
			igl::vertex_triangle_adjacency(m_V.rows(), m_F, Nei_F, F_V);
			compute_M(m_V, m_F, Nei_F, F_V, area, M);
			C_Matrix_fomulation(m_V, m_F, Nei_F, F_V, C_matrix);
			HN = M*(C_matrix * m_V);
			H = 0.5 * HN.rowwise().norm();
			//cout<< H;
			compute_Gaussian(m_V, m_F, area, Nei_F, F_V, Gaussian_C);
			m_H = 3*(H.array() - H.minCoeff()) / (H.maxCoeff() - H.minCoeff());
			igl::parula(m_H, false, m_C);
			m_G = 30*(Gaussian_C.array()) / (Gaussian_C.maxCoeff() - Gaussian_C.minCoeff());
			igl::parula(m_G, false, m_CG);
			if (Display == 1) {

				viewer.data().set_mesh(m_V, m_F);
				viewer.data().set_colors(m_C);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;
			}

			else if (Display == 2) {

				viewer.data().set_mesh(m_V, m_F);
				viewer.data().set_colors(m_CG);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;

			}
			else if (Display == 0) {
				viewer.data().set_mesh(m_V, m_F);
				viewer.data().show_overlay_depth = 1;
				viewer.data().show_overlay = 1;
			}
			/*
			Eigen::MatrixXd HN;
			Eigen::MatrixXd H;
			Eigen::SparseMatrix<double> L, M, Minv;
			igl::cotmatrix(m_V, m_F, L);
			igl::massmatrix(m_V, m_F, igl::MASSMATRIX_TYPE_VORONOI, M);
			igl::invert_diag(M, Minv);
			HN = -Minv * (L*m_V);
			H = HN.rowwise().norm(); //up to sign
			cout << H;
			*/
		}
		else if (mode == 2) {
			using namespace std;
			using namespace Eigen;
			using namespace Spectra;

			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			Eigen::MatrixXd area;
			Eigen::SparseMatrix<double> C_matrix;
			Eigen::SparseMatrix<double> C_matrix_ab;
			Eigen::SparseMatrix<double> M;
			Eigen::SparseMatrix<double> M_half;
			Eigen::SparseMatrix<double> M_Positive;
			Eigen::SparseMatrix<double> A;
			Eigen::MatrixXd eigen_value;
			Eigen::MatrixXd eigen_vector;
			Eigen::MatrixXd new_V;

			igl::vertex_triangle_adjacency(m_V.rows(), m_F, Nei_F, F_V);
			compute_M(m_V, m_F, Nei_F, F_V, area, M);
			C_Matrix_fomulation(m_V, m_F, Nei_F, F_V, C_matrix);
			Convert_M(M, M_half, M_Positive);
			A = M_half * C_matrix * M_half;
			//cout << A.row(0);
			double K;
			if (Display == 0) {
				K = 5;
			}
			else if (Display == 1) {
				K = 10;
			}
			else {
				K = 30;
			}
			SparseSymMatProd<double> op(A);
			SymEigsSolver< double, SMALLEST_MAGN, SparseSymMatProd<double> > eigs(&op, K, 400);
			eigs.init();
			int nconv = eigs.compute();

			// Retrieve results
			Eigen::VectorXd evalues;
			Eigen::MatrixXd e_vectors;
			if (eigs.info() == SUCCESSFUL)
				evalues = eigs.eigenvalues();
				e_vectors = eigs.eigenvectors();
			
			
			std::cout << "Eigenvalues found:\n" << evalues << std::endl;
			//std::cout << "Eigenvector found:\n" << e_vectors << std::endl;
			e_vectors = M_half * e_vectors;
			//std::cout << "Eigenvector found:\n" << e_vectors << std::endl;
			Formate_Eigen_V(m_V, M_Positive, e_vectors, new_V);
			viewer.data().set_mesh(new_V, m_F);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;

			//Eigen::EigenSolver<MatrixXd> es(A);
			//A = M * C_matrix;
			//Eigen::EigenSolver<MatrixXd> es(A);
			//cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
		}
		else if (mode == 3) {
			using namespace std;
			using namespace Eigen;

			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			Eigen::MatrixXd area;
			Eigen::MatrixXd Identity(m_V0.rows(), m_V0.rows());
			Eigen::SparseMatrix<double> C_matrix;
			Eigen::SparseMatrix<double> M;
			Eigen::SparseMatrix<double> L;
			Eigen::MatrixXd M_Dense;
			double lamda = 0.0000001;
			igl::vertex_triangle_adjacency(m_V5.rows(), m_F0, Nei_F, F_V);
			compute_M(m_V5, m_F0, Nei_F, F_V, area, M);
			C_Matrix_fomulation(m_V5, m_F0, Nei_F, F_V, C_matrix);
			L = M * C_matrix;
			for (int i = 0; i < m_V0.rows(); i++) {
				Identity(i, i) = 1;
			}
			//cout << Identity.row(1) << endl;
			m_V5 = (Identity + lamda * L) * m_V5;
			iter = iter + 1;
			if (iteration == 1) {
				m_V5 = m_V0;
				iter = 0;
			}
			cout << (L * m_V5).norm()<<" ";
			M_Dense = M.toDense();
			cout << "iteration: " << iter << endl;
			viewer.data().set_mesh(m_V5, m_F0);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;


		}		
		else if (mode == 4) {
			using namespace std;
			using namespace Eigen;

			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			Eigen::MatrixXd area;
			Eigen::SparseMatrix<double> C_matrix;
			Eigen::SparseMatrix<double> M;
			Eigen::SparseMatrix<double> M_half;
			Eigen::SparseMatrix<double> M_Positive;
			Eigen::SparseMatrix<double> L;
			Eigen::MatrixXd M_Dense;
			Eigen::SparseMatrix<double> Identity(m_V0.rows(), m_V0.rows());
			Identity.setIdentity();
			double lamda = 0.000001;
			igl::vertex_triangle_adjacency(m_V6.rows(), m_F0, Nei_F, F_V);
			compute_M(m_V6, m_F0, Nei_F, F_V, area, M);
			C_Matrix_fomulation(m_V6, m_F0, Nei_F, F_V, C_matrix);
			Convert_M(M, M_half, M_Positive);
			//cout << Identity.row(1) << endl;
			L = (M_Positive - lamda * C_matrix);
			Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
			solver.compute(L);
			m_V6 = solver.solve(M_Positive * m_V6);
			iter = iter + 1;
			if (iteration == 1) {
				m_V6 = m_V0;
				iter = 0;
			}
			cout << "iteration: " << iter << endl;
			viewer.data().set_mesh(m_V6, m_F0);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;
		}
		else {	

			using namespace std;
			using namespace Eigen;

			std::vector<vector<double> > Nei_F;
			std::vector<vector<double> > F_V;
			Eigen::MatrixXd area;
			Eigen::SparseMatrix<double> C_matrix;
			Eigen::SparseMatrix<double> M;
			Eigen::SparseMatrix<double> M_half;
			Eigen::SparseMatrix<double> M_Positive;
			Eigen::SparseMatrix<double> L;
			Eigen::MatrixXd M_Dense;
			Eigen::SparseMatrix<double> Identity(m_V0.rows(), m_V0.rows());
			Identity.setIdentity();
			double lamda = 0.000001;
			igl::vertex_triangle_adjacency(m_V7.rows(), m_F0, Nei_F, F_V);
			compute_M(m_V7, m_F0, Nei_F, F_V, area, M);
			C_Matrix_fomulation(m_V7, m_F0, Nei_F, F_V, C_matrix);
			Convert_M(M, M_half, M_Positive);
			//cout << Identity.row(1) << endl;
			L = (M_Positive - lamda * C_matrix);
			Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
			solver.compute(L);
			m_V7 = solver.solve(M_Positive * m_V7);
			iter = iter + 1;
			if (iteration == 1) {
				m_V7 = m_V70;
				iter = 0;
			}
			cout << "iteration: " << iter << endl;
			viewer.data().set_mesh(m_V7, m_F0);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;
			
		}
		
		//======================================================================			
	}
	
private:

};

MyContext g_myctx;


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key=='q' || key=='Q')
	{
		exit(0);
	}
	return false;
}


int main(int argc, char *argv[])
{
	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// menu variable Shared between two menus
	double doubleVariable = 0.1f; 

	// Add content to the default menu window via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

			// Add a button
			if (ImGui::Button("Print Hello", ImVec2(-1, 0)))
			{
				g_myctx.reset_display(viewer);
			}
		}
	};

	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "MyOperation", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		// point size
		// [event handle] if value changed

		// vertex index
		if (ImGui::SliderInt("Display", &g_myctx.Display, 0,2))
		{
			g_myctx.reset_display(viewer);
		} 
		//mode
		if (ImGui::SliderInt("Question", &g_myctx.mode, 0,5))
		{
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Button("iterate", ImVec2(-1, 0)))
		{
			g_myctx.iteration = 0;
			g_myctx.reset_display(viewer);
		}

		if (ImGui::Button("reset", ImVec2(-1, 0)))
		{
			g_myctx.iteration = 1;
			g_myctx.reset_display(viewer);
		}

		//mode-text
		if (g_myctx.mode==0)
		{
			ImGui::Text("Question 1");
		}
		else if (g_myctx.mode == 1) { 
			ImGui::Text("Question 2 ");
		}
		else if (g_myctx.mode == 2) {
			ImGui::Text("Question 3");
		}
		else if (g_myctx.mode == 3) {
			ImGui::Text("Question 5");
		}
		else if (g_myctx.mode == 4) {
			ImGui::Text("Question 6");
		}
		else if (g_myctx.mode == 5) {
			ImGui::Text("Question 7");
		}
		ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}
