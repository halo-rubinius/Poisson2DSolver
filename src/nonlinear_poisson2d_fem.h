#pragma once
#include <functional>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "node.h"
#include "triangular_element.h"

using namespace Eigen;

class nonlinear_poisson2d_fem {
public:
    typedef SparseMatrix<double> SpMat;

    NodeList nodes;
    std::vector<triangular_linear_element> elements;

    nonlinear_poisson2d_fem() : stiffness_from_operator_is_assembled(false) {}

    void set_initial_guess(
        std::function<double(double, double)> initial_guess_function);

    void set_source(std::function<double(double)> source,
                    std::function<double(double)> source_derivative);

    void solve();

private:
    SpMat global_stiffness_from_operator;
    SpMat global_stiffness_from_source;
    SpMat global_stiffness;
    VectorXd global_rhs;
    VectorXd global_rhs_from_operator;
    VectorXd global_rhs_from_source;
    VectorXd global_rhs_from_dirichlet;
    VectorXd solution;
    VectorXd initial_guess;
    std::function<double(double)> source_function, source_derivative_function;
    bool stiffness_from_operator_is_assembled;

    void assemble_stiffness_from_operator();

    void assemble_stiffness_from_source(const VectorXd& previous_solution);

    void assemble_stiffness(const VectorXd& previous_solution);

    void assmeble_rhs_from_operator(const VectorXd& previous_solution);

    void assmeble_rhs_from_source(const VectorXd& previous_solution);

    void prepare_system_of_equations(const VectorXd& previous_solution);

    void apply_boundary_conditions(const VectorXd& previous_solution);
};
