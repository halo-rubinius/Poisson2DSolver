#pragma once
#include <vector>

// 节点状态
enum class node_status {
    FREE,                // 表示节点为自由节点
    DIRICHLET_BOUNDARY,  // 表示节点为Dirichlet边界条件节点
    NEUMANN_BOUNDARY     // 表示节点为Neumann边界条件节点
};

// 节点
struct node {
public:
    double x, y;
    node_status status;  // 节点的状态

    /* 求解后，value表示节点的函数值；求解前，value的值由节点的status状态不同而具有不同的意义
    status为FREE时，value在求解前未定义，一般初始化为0.0
    status为DIRICHLET_BOUNDARY时，value在求解前表示Dirichlet边界条件值
    status为NEUMANN_BOUNDARY时，value在求解前表示Neumann边界条件值
    */
    double value;

    node(double x, double y)
        : x(x), y(y), status(node_status::FREE), value(0.0) {}

    node(double x, double y, node_status status, double value) {
        this->x = x;
        this->y = y;
        this->status = status;
        this->value = value;
    }
};

typedef std::vector<node> NodeList;  // 节点列表类型申明
