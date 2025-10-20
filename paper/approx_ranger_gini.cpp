#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
void approx_ranger_gini_cpp(List& child_nodeIDs, List& split_varIDs, List& split_values, 
                       DataFrame& x_old, DataFrame& x_new) {
  // Loop over trees
  for (int i = 0; i < child_nodeIDs.size(); i++) {
    int num_splits = as<IntegerVector>(as<List>(child_nodeIDs[i])[0]).size();
    
    // Loop over splits
    for (int j = 0; j < num_splits; j++) {
      int left_nodeID = as<IntegerVector>(as<List>(child_nodeIDs[i])[0])[j];
      //int right_nodeID = as<IntegerVector>(as<List>(child_nodeIDs[i])[1])[j];
      int split_varID = as<IntegerVector>(split_varIDs[i])[j];
      double split_value = as<NumericVector>(split_values[i])[j];
      
      // Don't do anything in leaves
      if (left_nodeID == 0) {
        continue;
      }
      
      // Get the split assignment of the original split in old data
      LogicalVector old_split_idx = as<NumericVector>(x_old[split_varID]) <= split_value;
      
      // Try all possible splits with the new data
      double best_smc = -1;
      for (int k = 0; k < x_new.size(); k++) {
        // Possible split values are the unique values of the new data
        NumericVector unique_values = unique(as<NumericVector>(x_new[k]));
        
        // Find the best split value 
        for (int l = 0; l < unique_values.size(); l++) {
          double new_split_value = unique_values[l];
          LogicalVector new_split_idx = as<NumericVector>(x_new[k]) <= new_split_value;

          LogicalVector inv_old_split_idx = !old_split_idx;
          LogicalVector inv_new_split_idx = !new_split_idx;
          
          double counts_11 = sum(as<NumericVector>((old_split_idx[new_split_idx])));
          double counts_01 = sum(as<NumericVector>((inv_old_split_idx[new_split_idx])));
          double counts_10 = sum(as<NumericVector>((old_split_idx[inv_new_split_idx])));
          double counts_00 = sum(as<NumericVector>((inv_old_split_idx[inv_new_split_idx])));
          
          double sum_left = counts_11 * counts_11 + counts_01 * counts_01;
          double sum_right = counts_10 * counts_10 + counts_00 * counts_00;
          
          double n_left = counts_11 + counts_01;
          double n_right = counts_10 + counts_00;
          
          double smc = sum_right / (double) n_right + sum_left / (double) n_left;
          
          // If better then before, overwrite the split
          if (smc > best_smc) {
            best_smc = smc;
            as<NumericVector>(split_varIDs[i])[j] = k;
            as<NumericVector>(split_values[i])[j] = new_split_value;
          }
        }
      }
    }
  }
}

// [[Rcpp::export]]
void approx_ranger_local_gini_cpp(List& child_nodeIDs, List& split_varIDs, List& split_values, 
                             NumericMatrix& x_old, NumericMatrix& x_new) {
  // Loop over trees
  for (int i = 0; i < child_nodeIDs.size(); i++) {
    int num_splits = as<IntegerVector>(as<List>(child_nodeIDs[i])[0]).size();
    
    // List with observation indices for each node
    LogicalMatrix idx(x_old.nrow(), num_splits);
    idx(_, 0) = !idx(_, 0);
    
    // Loop over nodes
    for (int j = 0; j < num_splits; j++) {
      int left_nodeID = as<IntegerVector>(as<List>(child_nodeIDs[i])[0])[j];
      int right_nodeID = as<IntegerVector>(as<List>(child_nodeIDs[i])[1])[j];
      int split_varID = as<IntegerVector>(split_varIDs[i])[j];
      double split_value = as<NumericVector>(split_values[i])[j];
      
      // Don't do anything in leaves
    if (left_nodeID == 0) {
      continue;
    }
    
    // Get the values of the observations in this node
    LogicalVector node_idx = idx(_, j);
    NumericVector old_values = x_old(_, split_varID);
    NumericVector old_node_values = old_values[node_idx];
    
    // Get the split assignment of the original split in old data
    LogicalVector old_split_idx = old_node_values <= split_value;
    
    // Assign the indices for the child nodes
    LogicalVector temp_idx_left(x_old.nrow());
    temp_idx_left[node_idx] = old_split_idx;
    
    LogicalVector temp_idx_right(x_old.nrow());
    LogicalVector inv_old_split_idx = !old_split_idx;
    temp_idx_right[node_idx] = inv_old_split_idx;
    
    idx(_, left_nodeID) = temp_idx_left;
    idx(_, right_nodeID) = temp_idx_right;
    
    // Try all possible splits with the new data
    double best_smc = -1;
    for (int k = 0; k < x_new.ncol(); k++) {
      NumericVector new_values = x_old(_, k);
      NumericVector new_node_values = new_values[node_idx];
      
      // Possible split values are the unique values of the new data
      NumericVector unique_values = unique(new_node_values);
      
      // Find the best split value 
      for (int l = 0; l < unique_values.size(); l++) {
        double new_split_value = unique_values[l];
        LogicalVector new_split_idx = new_node_values <= new_split_value;
        
        // Measure similarity between the two splits with the simple matching coefficient
        LogicalVector inv_old_split_idx = !old_split_idx;
        LogicalVector inv_new_split_idx = !new_split_idx;
        
        double counts_11 = sum(as<NumericVector>((old_split_idx[new_split_idx])));
        double counts_01 = sum(as<NumericVector>((inv_old_split_idx[new_split_idx])));
        double counts_10 = sum(as<NumericVector>((old_split_idx[inv_new_split_idx])));
        double counts_00 = sum(as<NumericVector>((inv_old_split_idx[inv_new_split_idx])));
        
        double sum_left = counts_11 * counts_11 + counts_01 * counts_01;
        double sum_right = counts_10 * counts_10 + counts_00 * counts_00;
        
        double n_left = counts_11 + counts_01;
        double n_right = counts_10 + counts_00;
        
        double smc = sum_right / (double) n_right + sum_left / (double) n_left;
        
        // If better then before, overwrite the split
        if (smc > best_smc) {
          best_smc = smc;
          as<NumericVector>(split_varIDs[i])[j] = k;
          as<NumericVector>(split_values[i])[j] = new_split_value;
        }
      }
    }
    }
  }
}

