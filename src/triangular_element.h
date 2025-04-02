#pragma once
#include <array>
#include <functional>
#include <vector>

#include "Eigen/Dense"
#include "gauss_quadrature.h"
#include "node.h"

using namespace Eigen;

// 三角形线性单元
class triangular_linear_element {
public:
    std::array<int, 3> node_indices;  // 单元的3个顶点编号，按逆时针顺序排列

    Matrix3d element_stiffness_from_operator(const NodeList& nodes);

    Matrix3d element_stiffness_from_nonlinear_source(
        std::function<double(double)> source_derivative_function,
        const VectorXd& previous_solution);

    Vector3d compute_rhs_from_operator(const Matrix3d& Ke,
                                       const VectorXd& previous_solution);

    Vector3d compute_rhs_from_nonlinear_source(
        std::function<double(double)> nonlinear_source_function,
        const VectorXd& previous_solution);

private:
    double _area;               // 单元面积
    Vector3d _xcoord, _ycoord;  // 单元顶点坐标
    static gauss_quadrature_triangle gauss_integrator;

    // 计算单元的面积
    inline void compute_area(const NodeList& nodes) {
        _area = 0.5 * ((_xcoord[1] - _xcoord[0]) * (_ycoord[2] - _ycoord[0]) -
                       (_xcoord[2] - _xcoord[0]) * (_ycoord[1] - _ycoord[0]));
    }

    // 获取顶点坐标
    inline void get_vertices_coordinates(const NodeList& nodes) {
        for (int i = 0; i < 3; i++) {
            _xcoord[i] = nodes[node_indices[i]].x;
            _ycoord[i] = nodes[node_indices[i]].y;
        }
    }

    static double apply_shape_function(const Vector2d& natural_coordinate,
                                       const Vector3d& t) {
        Vector3d shape(1.0 - natural_coordinate[0] - natural_coordinate[1],
                       natural_coordinate[0], natural_coordinate[1]);
        return shape.dot(t);
    }

    static Vector3d shape_function(const Vector2d& natural_coordinate) {
        return Vector3d(1.0 - natural_coordinate[0] - natural_coordinate[1],
                        natural_coordinate[0], natural_coordinate[1]);
    }
};
