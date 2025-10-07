#ifndef KDTREE_H
#define KDTREE_H

#define K_DIM 1280

struct Point {
    int id;                
    int label;             
    double vec[K_DIM];     
};

struct KDNode {
    struct Point pt;       
    struct KDNode *left;   
    struct KDNode *right;  
    int axis;              
};


struct KDNode* build_kdtree(struct Point points[], int n, int depth);

void nearest_neighbor_search(struct KDNode* root, struct Point* target, struct KDNode** best_node, double* best_dist);

void free_kdtree(struct KDNode* node);

#endif 