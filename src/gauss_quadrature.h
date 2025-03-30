#pragma once
#include <functional>
#include <vector>

#include "Eigen/Dense"

using namespace Eigen;

// 计算在标准三角形单元上的Gauss-Legendre数值积分
class gauss_quadrature_triangle {
public:
    std::vector<double> gauss_weights;
    std::vector<Vector2d> gauss_points;

    size_t size() const { return gauss_weights.size(); }

    gauss_quadrature_triangle() {
        constexpr int M = 10;
        double standard_gauss_points[M] = {
            -0.1488743389816312, 0.1488743389816312,  -0.4333953941292472,
            0.4333953941292472,  -0.6794095682990244, 0.6794095682990244,
            -0.8650633666889845, 0.8650633666889845,  -0.9739065285171717,
            0.9739065285171717};

        double standard_gauss_weights[M] = {
            0.2955242247147529, 0.2955242247147529, 0.2692667193099963,
            0.2692667193099963, 0.2190863625159820, 0.2190863625159820,
            0.1494513491505806, 0.1494513491505806, 0.0666713443086881,
            0.0666713443086881};

        gauss_weights.clear();
        gauss_points.clear();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                gauss_weights.emplace_back((1.0 + standard_gauss_points[i]) /
                                           8.0 * standard_gauss_weights[i] *
                                           standard_gauss_weights[j]);
                double pr = (1.0 + standard_gauss_points[i]) *
                            (1.0 + standard_gauss_points[j]) / 4.0;
                double qr = (1.0 + standard_gauss_points[i]) *
                            (1.0 - standard_gauss_points[j]) / 4.0;
                gauss_points.emplace_back(Vector2d(pr, qr));
            }
        }
    }

    double integrate(std::function<double(double, double)> integrand_function) {
        double result = 0.0;
        for (int i = 0; i < gauss_weights.size(); i++) {
            result += gauss_weights[i] * integrand_function(gauss_points[i][0],
                                                            gauss_points[i][1]);
        }
        return result;
    }
};
