#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/kdtree.h"


static int sort_axis;

double distance_squared(struct Point* p1, struct Point* p2) {
    double total = 0.0;
    for (int i = 0; i < K_DIM; i++) {
        double diff = p1->vec[i] - p2->vec[i];
        total += diff * diff;
    }
    return total;
}

int compare_points(const void* a, const void* b) {
    struct Point* p1 = (struct Point*)a;
    struct Point* p2 = (struct Point*)b;
    if (p1->vec[sort_axis] < p2->vec[sort_axis]) return -1;
    if (p1->vec[sort_axis] > p2->vec[sort_axis]) return 1;
    return 0;
}

struct KDNode* build_kdtree(struct Point points[], int n, int depth) {
    if (n <= 0) {
        return NULL;
    }

    int axis = depth % K_DIM;
    sort_axis = axis;

    qsort(points, n, sizeof(struct Point), compare_points);
    int median_idx = n / 2;

    struct KDNode* node = (struct KDNode*)malloc(sizeof(struct KDNode));
    if (!node) {
        perror("Failed to allocate node");
        exit(EXIT_FAILURE);
    }
    node->pt = points[median_idx];
    node->axis = axis;

    node->left = build_kdtree(points, median_idx, depth + 1);

    node->right = build_kdtree(points + median_idx + 1, n - median_idx - 1, depth + 1);

    return node;
}


void nearest_neighbor_search(struct KDNode* root, struct Point* target, struct KDNode** best_node, double* best_dist) {
    if (root == NULL) {
        return;
    }

    double d_sq = distance_squared(target, &root->pt);
    if (d_sq < *best_dist && root->pt.id != target->id) {
        *best_dist = d_sq;
        *best_node = root;
    }

    int axis = root->axis;
    double diff = target->vec[axis] - root->pt.vec[axis];
    
    struct KDNode *near_branch;
    struct KDNode *far_branch;

    if (diff < 0) {
        near_branch = root->left;
        far_branch = root->right;
    } else {
        near_branch = root->right;
        far_branch = root->left;
    }

    nearest_neighbor_search(near_branch, target, best_node, best_dist);


    if (diff * diff < *best_dist) {
        nearest_neighbor_search(far_branch, target, best_node, best_dist);
    }
}

void free_kdtree(struct KDNode* node) {
    if (node == NULL) {
        return;
    }
    free_kdtree(node->left);
    free_kdtree(node->right);
    free(node);
}