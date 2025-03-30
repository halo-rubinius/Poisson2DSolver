#include "triangular_element.h"

Matrix3d triangular_linear_element::element_stiffness_from_operator(
    const NodeList& nodes) {
    get_vertices_coordinates(nodes);
    compute_area(nodes);

    double x12 = _xcoord[0] - _xcoord[1];
    double x23 = _xcoord[1] - _xcoord[2];
    double x31 = _xcoord[2] - _xcoord[0];
    double y12 = _ycoord[0] - _ycoord[1];
    double y23 = _ycoord[1] - _ycoord[2];
    double y31 = _ycoord[2] - _ycoord[0];

    double area4 = -1.0 / (4.0 * this->_area);

    Matrix3d Ke;  // 单元刚度矩阵
    Ke(0, 0) = (y23 * y23 + x23 * x23) * area4;
    Ke(0, 1) = Ke(1, 0) = (y23 * y31 + x23 * x31) * area4;
    Ke(0, 2) = Ke(2, 0) = (y23 * y12 + x23 * x12) * area4;
    Ke(1, 1) = (y31 * y31 + x31 * x31) * area4;
    Ke(1, 2) = Ke(2, 1) = (y31 * y12 + x31 * x12) * area4;
    Ke(2, 2) = (y12 * y12 + x12 * x12) * area4;
    return Ke;
}

Matrix3d triangular_linear_element::element_stiffness_from_nonlinear_source(
    std::function<double(double)> source_derivative_function,
    const VectorXd& previous_solution) {
    Vector3d solution;
    solution[0] = previous_solution[node_indices[0]];
    solution[1] = previous_solution[node_indices[1]];
    solution[2] = previous_solution[node_indices[2]];

    Matrix3d Kf = Matrix3d::Zero();

    for (int k = 0; k < gauss_integrator.gauss_weights.size(); k++) {
        Vector3d shape_function_of_gauss_point =
            shape_function(gauss_integrator.gauss_points[k]);
        double u = shape_function_of_gauss_point.dot(solution);
        double c =
            gauss_integrator.gauss_weights[k] * source_derivative_function(u);
        Kf += c * shape_function_of_gauss_point *
              shape_function_of_gauss_point.transpose();
    }

    Kf *= (-2.0 * this->_area);
    return Kf;
}

Vector3d triangular_linear_element::compute_rhs_from_nonlinear_source(
    std::function<double(double)> nonlinear_source_function,
    const VectorXd& previous_solution) {
    Vector3d solution, bf = Vector3d::Zero();
    solution[0] = previous_solution[node_indices[0]];
    solution[1] = previous_solution[node_indices[1]];
    solution[2] = previous_solution[node_indices[2]];

    for (int k = 0; k < gauss_integrator.gauss_weights.size(); k++) {
        Vector3d shape_function_of_gauss_point =
            shape_function(gauss_integrator.gauss_points[k]);
        double u = shape_function_of_gauss_point.dot(solution);
        double c =
            gauss_integrator.gauss_weights[k] * nonlinear_source_function(u);
        bf += c * shape_function_of_gauss_point;
    }

    bf *= (2.0 * _area);
    return bf;
}

Vector3d triangular_linear_element::compute_rhs_from_operator(
    const Matrix3d& Ke, const VectorXd& previous_solution) {
    Vector3d solution;
    solution[0] = previous_solution[node_indices[0]];
    solution[1] = previous_solution[node_indices[1]];
    solution[2] = previous_solution[node_indices[2]];

    return Ke * solution;
}

gauss_quadrature_triangle triangular_linear_element::gauss_integrator =
    gauss_quadrature_triangle();
