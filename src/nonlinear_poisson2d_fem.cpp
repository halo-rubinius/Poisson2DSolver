#include "nonlinear_poisson2d_fem.h"

#include <iostream>

#include "Eigen/Dense"
#include "Eigen/Sparse"

using namespace Eigen;

void nonlinear_poisson2d_fem::set_initial_guess(
    std::function<double(double, double)> initial_guess_function) {
    initial_guess = VectorXd(nodes.size());

    for (int i = 0; i < nodes.size(); i++) {
        initial_guess[i] = initial_guess_function(nodes[i].x, nodes[i].y);
    }
}

void nonlinear_poisson2d_fem::set_source(
    std::function<double(double)> source,
    std::function<double(double)> source_derivative) {
    this->source_function = source;
    this->source_derivative_function = source_derivative;
}

void nonlinear_poisson2d_fem::assemble_stiffness_from_operator() {
    std::vector<Triplet<double>> triplets;

    for (auto& element : elements) {
        Matrix3d Ke = element.element_stiffness_from_operator(nodes);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                triplets.emplace_back(element.node_indices[i],
                                      element.node_indices[j], Ke(i, j));
            }
        }
    }
    global_stiffness_from_operator = SpMat(nodes.size(), nodes.size());
    global_stiffness_from_operator.setFromTriplets(triplets.begin(),
                                                   triplets.end());
    stiffness_from_operator_is_assembled = true;
}

void nonlinear_poisson2d_fem::assemble_stiffness_from_source(
    const VectorXd& previous_solution) {
    std::vector<Triplet<double>> triplets;

    for (auto& element : elements) {
        Matrix3d Kf = element.element_stiffness_from_nonlinear_source(
            source_derivative_function, previous_solution);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                triplets.emplace_back(element.node_indices[i],
                                      element.node_indices[j], Kf(i, j));
            }
        }
    }
    global_stiffness_from_source = SpMat(nodes.size(), nodes.size());
    global_stiffness_from_source.setFromTriplets(triplets.begin(),
                                                 triplets.end());
}

void nonlinear_poisson2d_fem::assemble_stiffness(
    const VectorXd& previous_solution) {
    global_rhs_from_dirichlet = VectorXd::Zero(nodes.size());

    std::vector<Triplet<double>> triplets;
    for (auto& element : elements) {
        Matrix3d Ke = element.element_stiffness_from_operator(nodes) +
                      element.element_stiffness_from_nonlinear_source(
                          source_derivative_function, previous_solution);
        for (int i = 0; i < 3; i++) {
            int i_node = element.node_indices[i];
            if (nodes[i_node].status == node_status::DIRICHLET_BOUNDARY) {
                triplets.emplace_back(i_node, i_node, Ke(i, i));
            } else {
                for (int j = 0; j < 3; j++) {
                    int j_node = element.node_indices[j];
                    if (nodes[j_node].status ==
                        node_status::DIRICHLET_BOUNDARY) {
                        global_rhs_from_dirichlet[i_node] +=
                            -Ke(i, j) *
                            (nodes[j_node].value - previous_solution[j_node]);
                    } else {
                        triplets.emplace_back(i_node, j_node, Ke(i, j));
                    }
                }
            }
        }
    }
    global_stiffness = SpMat(nodes.size(), nodes.size());
    global_stiffness.setFromTriplets(triplets.begin(), triplets.end());
}

void nonlinear_poisson2d_fem::assmeble_rhs_from_operator(
    const VectorXd& previous_solution) {
    global_rhs_from_operator =
        global_stiffness_from_operator * previous_solution;
}

void nonlinear_poisson2d_fem::assmeble_rhs_from_source(
    const VectorXd& previous_solution) {
    Vector3d bf;

    global_rhs_from_source = VectorXd::Zero(nodes.size());

    for (auto& element : elements) {
        bf = element.compute_rhs_from_nonlinear_source(source_function,
                                                       previous_solution);
        for (int i = 0; i < 3; i++) {
            global_rhs_from_source[element.node_indices[i]] += bf[i];
        }
    }
}

void nonlinear_poisson2d_fem::prepare_system_of_equations(
    const VectorXd& previous_solution) {
    assemble_stiffness(previous_solution);

    if (!stiffness_from_operator_is_assembled) {
        assemble_stiffness_from_operator();
    }
    assmeble_rhs_from_operator(previous_solution);
    assmeble_rhs_from_source(previous_solution);
    global_rhs = global_rhs_from_operator + global_rhs_from_source;

    apply_boundary_conditions(previous_solution);
}

void nonlinear_poisson2d_fem::apply_boundary_conditions(
    const VectorXd& previous_solution) {
    global_rhs += global_rhs_from_dirichlet;
    for (int i = 0; i < nodes.size(); i++) {
        if (nodes[i].status == node_status::DIRICHLET_BOUNDARY) {
            global_stiffness.coeffRef(i, i) = 1.0;
            global_rhs[i] = nodes[i].value - previous_solution[i];
        }
    }
}

void nonlinear_poisson2d_fem::solve() {
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg_solver;
    cg_solver.setMaxIterations(2000);
    cg_solver.setTolerance(1.0e-8);

    double rel_tol = 1.0e-6;
    double abs_tol = 1.0e-6;

    VectorXd solution_updates, previous_solution = initial_guess;

    constexpr int MAX_NEWTON_ITERATION = 20;
    for (int newton_iteration = 0; newton_iteration < MAX_NEWTON_ITERATION;
         newton_iteration++) {
        prepare_system_of_equations(previous_solution);

        cg_solver.compute(global_stiffness);

        solution_updates = cg_solver.solve(global_rhs);
        solution = previous_solution + solution_updates;

        double abs_error = global_rhs.norm();
        double rel_error = solution_updates.norm() / previous_solution.norm();

        std::cout << "Newton iteration: " << newton_iteration << "  "
                  << abs_error << "  " << rel_error << std::endl;

        if (rel_error < rel_tol && abs_error < abs_tol) {
            break;
        }

        previous_solution = solution;
    }
}
