#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>     
#include <float.h> 
#include "../include/kdtree.h"

#define NUM_POINTS 50000
#define FILENAME "data/vectors.csv"

struct Point* load_points_from_csv(const char* filename, int num_points) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Failed to open file");
        return NULL;
    }

    struct Point* points = (struct Point*)malloc(num_points * sizeof(struct Point));
    if (points == NULL) {
        perror("Failed to allocate memory for points");
        fclose(fp);
        return NULL;
    }

    char line[30000]; 
    for (int i = 0; i < num_points; i++) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            fprintf(stderr, "Error: File ended prematurely at line %d\n", i + 1);
            free(points); 
            fclose(fp);
            return NULL;
        }
        points[i].id = i;

        char* token = strtok(line, ",");
        points[i].label = atoi(token);

        for (int j = 0; j < K_DIM; j++) {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Error: Incomplete vector at line %d\n", i + 1);
                free(points); 
                fclose(fp);
                return NULL;
            }
            points[i].vec[j] = atof(token);
        }
    }

    fclose(fp);
    return points;
}


int main() {
    printf("Loading data from %s...\n", FILENAME);
    struct Point* points = load_points_from_csv(FILENAME, NUM_POINTS);
    if (points == NULL) {
        return 1; 
    }
    printf("Data loaded successfully. %d points in memory.\n", NUM_POINTS);

    printf("Building k-d tree...\n");
    struct KDNode* root = build_kdtree(points, NUM_POINTS, 0);
    printf("k-d tree built successfully.\n\n");

    while (1) {
        int query_id;
        printf("Enter an image ID (0-%d) to find its nearest neighbor (or -1 to exit): ", NUM_POINTS - 1);
        
        if (scanf("%d", &query_id) != 1) {
            while(getchar() != '\n'); 
            printf("Invalid input. Please enter a number.\n");
            continue;
        }

        if (query_id == -1) break;

        if (query_id < 0 || query_id >= NUM_POINTS) {
            printf("Invalid ID. Please enter a number between 0 and %d.\n", NUM_POINTS - 1);
            continue;
        }
        
        struct Point* target = NULL;
        for (int i = 0; i < NUM_POINTS; i++) {
            if (points[i].id == query_id) {
                target = &points[i];
                break;
            }
        }
        
        if (target == NULL) {
             printf("Error: Could not find point with ID %d.\n", query_id);
             continue;
        }

        struct KDNode* best_node = NULL;
        double best_dist = DBL_MAX;

        nearest_neighbor_search(root, target, &best_node, &best_dist);

        printf("\n--- Search Results ---\n");
        printf("Query Image:  ID=%d, Label=%d\n", target->id, target->label);
        if (best_node != NULL) {
            printf("Found Match:  ID=%d, Label=%d\n", best_node->pt.id, best_node->pt.label);
            printf("Distance:     %.4f\n\n", sqrt(best_dist));
        } else {
            printf("No match found for a different image.\n\n");
        }
    }

    printf("Freeing k-d tree and exiting...\n");
    free_kdtree(root);
    free(points);

    return 0;
}