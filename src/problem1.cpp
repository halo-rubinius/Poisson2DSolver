#include "problem1.h"

#include <cmath>

double source_func(double u) { return exp(-u); }

double source_derivative(double u) { return -exp(-u); }

double initial_guess(double x, double y) { return exp(-x - y); }

void problem_1::solve() {
    generate_mesh();

    solver.set_initial_guess(initial_guess);

    solver.set_source(source_func, source_derivative);

    solver.solve();
}

void problem_1::generate_mesh() {
    int node_id = 0;
    double dx = _Lx / _Nx;
    double dy = _Ly / _Ny;

    solver.nodes.clear();
    for (int iy = 0; iy < _Ny + 1; iy++) {
        for (int ix = 0; ix < _Nx + 1; ix++) {
            node_id = ix + iy * (_Nx + 1);
            if (ix > 0 && ix < _Nx) {
                solver.nodes.emplace_back(node(ix * dx, iy * dy));
            } else {
                if (ix == 0) {
                    solver.nodes.emplace_back(
                        node(ix * dx, iy * dy, node_status::DIRICHLET_BOUNDARY,
                             2.0));
                    continue;
                }
                if (ix == _Nx) {
                    solver.nodes.emplace_back(
                        node(ix * dx, iy * dy, node_status::DIRICHLET_BOUNDARY,
                             1.0));
                    continue;
                }
            }
        }
    }

    triangular_linear_element element;
    solver.elements.clear();
    for (int iy = 0; iy < _Ny; iy++) {
        for (int ix = 0; ix < _Nx; ix++) {
            node_id = (_Nx + 1) * iy + ix;
            element.node_indices = {node_id, node_id + 1, node_id + _Nx + 2};
            solver.elements.emplace_back(element);
            element.node_indices = {node_id, node_id + _Nx + 2,
                                    node_id + _Nx + 1};
            solver.elements.emplace_back(element);
        }
    }
}
