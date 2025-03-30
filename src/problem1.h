#pragma once

#include "nonlinear_poisson2d_fem.h"

class problem_1 {
public:
    problem_1(double Lx, double Ly, int Nx, int Ny)
        : _Lx(Lx), _Ly(Ly), _Nx(Nx), _Ny(Ny) {}

    void solve();

private:
    double _Lx, _Ly;
    int _Nx, _Ny;
    nonlinear_poisson2d_fem solver;

    void generate_mesh();
};
