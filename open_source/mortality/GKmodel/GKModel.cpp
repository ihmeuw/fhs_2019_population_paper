#include <TMB.hpp>
using namespace density;
using Eigen::SparseMatrix;


// UTILITY FUNCTIONS

// NAME:    test_print
// DESC:    prints a string if testing only though
// IN:      char
// OUT:     None
int test_print(char* statement, int testing_prints){
    if (testing_prints == 1){
        printf("%s\n", statement);
    }
    return 1;
}

// NAME:    expit
// DESC:    takes a number on logit scale and returns it to the zero to
//          one number line
// IN:      Type
// OUT:     Type
template<class Type>
Type expit(Type logit_param){
    Type param = exp(logit_param) / (1 + exp(logit_param));
    return param;
}


// NAME:    non_zero
// DESC:    returns a size_t object but turns ones to zeros
// IN:      size_t
// OUT:     size_t
size_t non_zero(size_t d) {
    size_t r;
    if (d > 0){
        r = d;
    }
    else{
        r = 1;
    }
    return r;
}

// NAME:    param_index
// DESC:    returns the index to use for a parameter based of length of its
//          deminsion and the index of the covariate
// IN:      size_t, size_t
// OUT:     size_t
size_t param_index(size_t dim_size, size_t index){
    size_t index_new;
    if (dim_size > 1){
        index_new = index;
    }
    else{
        index_new = 0;
    }
    return index_new;
}

// NAME:    replace_nan
// DESC:    replaces a nan with 0 else returns original object
// IN:      Type
// OUT:     Type
template<class Type>
Type replace_nan(Type nll_add){
    Type new_nll_add;
    if (CppAD::isnan(nll_add)){
        new_nll_add = 0.;
    }
    else{
        new_nll_add = nll_add;
    }
    return new_nll_add;
}

// NAME:    zero_3d
// DESC:    return a 3d array with the same shape but zeros in all elements
// IN:      array<Type>
// OUT:     array<Type>
template<class Type>
array<Type> zero_3d(array<Type> arr) {
    size_t L = arr.dim(0);        // number of locations
    size_t A = arr.dim(1);        // number of ages
    size_t T = arr.dim(2);        // number of years
    array<Type> new_preds(L,A,T);
    for (size_t l = 0; l < L; l++) {
        for (size_t a = 0; a < A; a++) {
            for (size_t t = 0; t < T; t++) {
                new_preds(l,a,t) = Type(0.);
            }
        }
    }
    return new_preds;
}

// NAME:    mult_model
// DESC:    multiplies together two factors in a linear model based on their
//          dimension size
// IN:      array<Type>, array<Type>
// OUT:     array<Type>
template<class Type>
array<Type> mult_model(array<Type> cov, array<Type> parameter, size_t true_k, array<Type> zero_replace) {
    if (true_k == 0){
        return zero_replace;
    }
    size_t L = cov.dim(0);        // number of locations
    size_t A = cov.dim(1);        // number of ages
    size_t T = cov.dim(2);        // number of years
    size_t K = cov.dim(3);        // number of covariates
    size_t L_par = parameter.dim(0);        // number of locations par
    size_t A_par = parameter.dim(1);        // number of ages par
    size_t T_par = parameter.dim(2);        // number of years par
    size_t K_par = parameter.dim(3);        // number of covariates par
    array<Type> pred_sub(L,A,T);
    size_t l_;
    size_t a_;
    size_t t_;
    Type betaX;
    for (size_t l = 0; l < L; l++) {
        for (size_t a = 0; a < A; a++) {
            for (size_t t = 0; t < T; t++) {
                betaX = 0.;
                l_ = param_index(L_par, l);
                a_ = param_index(A_par, a);
                t_ = param_index(T_par, t);
                for (size_t k = 0; k < K; k++){
                    betaX += parameter(l_,a_,t_,k) * cov(l,a,t,k);
                }
                pred_sub(l,a,t) = betaX;
            }
        }
    }
    return pred_sub;
}

template<class Type>
array<Type> mult_model_region(array<Type> cov, array<Type> parameter, size_t true_k, array<Type> zero_replace, vector<int> region) {
    if (true_k == 0){
        return zero_replace;
    }
    size_t L = cov.dim(0);        // number of locations
    size_t A = cov.dim(1);        // number of ages
    size_t T = cov.dim(2);        // number of years
    size_t K = cov.dim(3);        // number of covariates
    size_t L_par = parameter.dim(0);        // number of locations par
    size_t A_par = parameter.dim(1);        // number of ages par
    size_t T_par = parameter.dim(2);        // number of years par
    size_t K_par = parameter.dim(3);        // number of covariates par
    array<Type> pred_sub(L,A,T);
    size_t l_;
    size_t a_;
    size_t t_;
    Type betaX;
    for (size_t l = 0; l < L; l++) {
        for (size_t a = 0; a < A; a++) {
            for (size_t t = 0; t < T; t++) {
                betaX = 0.;
                l_ = region[l];
                a_ = param_index(A_par, a);
                t_ = param_index(T_par, t);
                for (size_t k = 0; k < K; k++){
                    betaX += parameter(l_,a_,t_,k) * cov(l,a,t,k);
                }
                pred_sub(l,a,t) = betaX;
            }
        }
    }
    return pred_sub;
}

// NAME:    adjust_preds
// DESC:    adjust the predictions of a model to reflect the dat points
//          the total parameter determines if the transformation adds (0)
//          or subtracts (1) the mean adjust is 0 the preds array is returned.
// IN:      array<Type>, matrix<Type>, size_t, size_t
// OUT:     array<Type>
template<class Type>
array<Type> adjust_preds(array<Type> preds, matrix<Type> mean_mat, size_t total, size_t adjust) {
    if (adjust == 0){
        return preds;
    }
    size_t L = preds.dim(0);        // number of locations
    size_t A = preds.dim(1);        // number of ages
    size_t T = preds.dim(2);        // number of years
    array<Type> new_preds(L,A,T);
    for (size_t l = 0; l < L; l++) {
        for (size_t a = 0; a < A; a++) {
            for (size_t t = 0; t < T; t++) {
                if (total == 1){
                    new_preds(l,a,t) = preds(l,a,t) - mean_mat(l,a);
                }
                else{
                    new_preds(l,a,t) = preds(l,a,t) + mean_mat(l,a);
                }
            }
        }
    }
    return new_preds;
}

// NAME:    transform beta
// DESC:    returns a transformed beta based off the constraint
// IN:      Type, Type
// OUT:     Type
template<class Type>
Type transform_beta_value(Type b, Type c) {
    Type r;
    if (c == 1){
        r = exp(b);
    }
    else if (c == -1){
        r = Type(-1.) * exp(b);
    }
    else {
        r = b;
    }
    return r;
}

// NAME:    transform beta
// DESC:    returns a transformed beta array based off the constraint vector
// IN:      array<Type>, vector<Type>
// OUT:     array<Type>
template<class Type>
array<Type> transform_beta(array<Type> beta_raw, vector<Type> constraint) {
    size_t L_sub = beta_raw.dim(0);        // number of locations
    size_t A_sub = beta_raw.dim(1);        // number of ages
    size_t T_sub = beta_raw.dim(2);        // number of years
    size_t K_sub = beta_raw.dim(3);        // number of covariates
    array<Type> beta(non_zero(L_sub), non_zero(A_sub), non_zero(T_sub), non_zero(K_sub));
    for (size_t l = 0; l < L_sub; l++){
        for (size_t a = 0; a < A_sub; a++){
            for (size_t t = 0; t < T_sub; t++){
                for (size_t k = 0; k < K_sub; k++){
                    beta(l,a,t,k) = transform_beta_value(beta_raw(l,a,t,k), constraint[k]);
                }
            }
        }
    }
    return beta;
}

// NAME:    transform parameter
// DESC:    returns a transformed beta array based off the constraint vector
// IN:      array<Type>, Type
// OUT:     array<Type>
template<class Type>
array<Type> transform_parameter(array<Type> parameter_raw, Type constraint) {
    size_t L_sub = parameter_raw.dim(0);        // number of locations
    size_t A_sub = parameter_raw.dim(1);        // number of ages
    size_t T_sub = parameter_raw.dim(2);        // number of years
    size_t K_sub = parameter_raw.dim(3);
    array<Type> parameter(non_zero(L_sub), non_zero(A_sub), non_zero(T_sub), non_zero(K_sub));
    for (size_t l = 0; l < L_sub; l++){
        for (size_t a = 0; a < A_sub; a++){
            for (size_t t = 0; t < T_sub; t++){
                for (size_t k = 0; k < K_sub; k++){
                    parameter(l,a,t,k) = transform_beta_value(parameter_raw(l,a,t,k), constraint);
                }
            }
        }
    }
    return parameter;
}


// NAME:    re_prior
// DESC:    applies standard random effects priors
// IN:      Type, Type
// OUT:     Type
template<class Type>
Type re_prior(array<Type> gamma_sub, array<Type> tau) {
    size_t L_sub = gamma_sub.dim(0);        // number of locations
    size_t A_sub = gamma_sub.dim(1);        // number of ages
    size_t T_sub = gamma_sub.dim(2);        // number of years
    size_t Q_sub = gamma_sub.dim(3);        // number of covariates
    Type nll_sub = 0;
    for (size_t l = 0; l < L_sub; l++){
        for (size_t a = 0; a < A_sub; a++){
            for (size_t t = 0; t < T_sub; t++){
                for (size_t q = 0; q < Q_sub; q++){
                    nll_sub += replace_nan(dnorm(gamma_sub(l,a,t,q), Type(0.), tau(0,0,0,q), true));
                }
            }
        }
    }
    for (size_t q = 0; q < Q_sub; q++){
        nll_sub += replace_nan(dgamma(tau(0,0,0,q), Type(.1), Type(10.), true));
    }
    return nll_sub;
}

// NAME:    first_diff
// DESC:    returns a matrix which can be used to compute the vector of first differences of another vector.
// SOURCE:  https://github.com/IQSS/YourCast/blob/396408b87c032bc7588f28c75d3ea9801f8e1b93/R/make.priors.R#L604
// IN:      d, scalar, dimension of the vector for which we want to compute the first differences
// OUT:     (d-1) x d matrix of the form:
//             -1    1    0    0    0  ...
//              0   -1    1    0    0  ...
//              0    0   -1    1    0  ...
//              0    0    0   -1    1  ...
//             ...  ...  ...  ...  ... ...
template<class Type>
matrix<Type> first_diff(int d) {
    matrix<Type> M(d-1,d);
    for (size_t i = 0; i < d-1; i++) {
        for (size_t j = 0; j < d; j++) {
            // -1 on the diagonal
            if (i == j) M(i,j) = -1.;
            // +1 on the upper off-diagonal
            else if (i+1 == j) M(i,j) = 1.;
            // 0 everywhere else
            else M(i,j) = 0.;
        }
    }
    return M;
}

// NAME:    n_diff
// DESC:    returns a matrix which can be used to compute the vector of n-th differences of another vector.
// SOURCE:  https://github.com/IQSS/YourCast/blob/396408b87c032bc7588f28c75d3ea9801f8e1b93/R/make.priors.R#L658
// IN:      n, scalar
//          d, scalar, dimension of the vector for which we want to compute the first differences
// OUT:     (d-n) x d matrix. It is simply the matrix obtained by iterating the first difference operator n times.
//          It resembles the discretization of the $n$-th derivative and can be used to define smoothness functionals.
template<class Type>
matrix<Type> n_diff(int n, int d) {
    matrix<Type> M = first_diff<Type>(d);
    // multiply first differences for each higher order
    if (n > 1)
        for (size_t i = 1; i < n; i++)
            M = first_diff<Type>(M.rows()) * M;
    return M;
}

// NAME:    deriv
// DESC:    returns the n-th difference matrix with rows added to ensure conformability when n > 1
// SOURCE:  https://github.com/IQSS/YourCast/blob/396408b87c032bc7588f28c75d3ea9801f8e1b93/R/make.priors.R#L740
// IN:      n, scalar
//          d, scalar, dimension of the vector for which we want to compute the first differences
// OUT:     d x d matrix. It is simply the matrix obtained by iterating the first difference operator n times.
//          First/last rows are repeated to conform to d x d when n > 1
template<class Type>
matrix<Type> deriv(int n, int d) {

    matrix<Type> M(d,d);
    M.setZero();

    // simply return I if n==0
    if (n == 0) {
        for (size_t i = 0; i < d; i++)
            M(i,i) = 1.;
    }

    // otherwise find n_diff and add requisite rows to make it conformable
    else {

        // find n_diff
        matrix<Type> D = n_diff<Type>(n, d);

        // for n == 1, repeat the first row
        if (n == 1)
            M.row(0) = D.row(0);

        // for n >= 2, copy the first and last rows in n/2 times each
        if (n >= 2) {
            for (size_t i = 0; i < n/2; i++) {
                M.row(i) = D.row(0);
                M.row(d-(i+1)) = D.row(D.rows()-1);
            }
            // for odd n, repeat the first row
            if (n % 2 == 1) M.row(ceil(n/2)) = D.row(0);
        }

        // fill in the remaining rows from D
        for (size_t i = 0; i < D.rows(); i++)
            M.row(ceil(n/2) + i + (n % 2)) = D.row(i);
    }
    return M;
}

// NAME:    derivative_prior
// DESC:    returns the matrix W that defines a mixed smoothness prior
// SOURCE:  https://github.com/IQSS/YourCast/blob/396408b87c032bc7588f28c75d3ea9801f8e1b93/R/make.priors.R#L763
// IN:      d: scalar
//              dimension of the desired smoothing matrix
//          der_v: vector
//              weight of each derivative in the mixed smoothness function,
//              starting from the derivative of order 0 (the identity operator).
//                  Example 1: der_v = c(0,0,1) corresponds to a smoothness functional which
//                      penalizes the 2nd derivative (0*identity + 0*1st derivative + 1*2nd derivative)
//                  Example 2: der_v = c(0,1,1) corresponds to a smoothness functional
//                      which penalizes equally the 1st and 2nd derivative
//                      (0*identity + 1*1st derivative + 1*2nd derivative)
// OUT:     W: d x d matrix, the matrix which defines the smoothness prior.
template<class Type>
matrix<Type> derivative_prior(int d, vector<Type> der_v) {
    // start with zeros
    matrix<Type> W(d,d);
    W.setZero();

    // find weights (see YourCast for possible extensions - for now, just weight everything equally)
    // should be diagonal 1/d
    matrix<Type> weights(d,d);
    weights.setZero();
    for (size_t i = 0; i < d; i++)
        weights(i,i) = 1. / d;

    // add on the weights for each derivative included in the prior
    if (d > 2){
        for (size_t i = 0; i < der_v.size(); i++) {
            if (der_v(i) != 0.) {
                matrix<Type> m = deriv<Type>(i, d);
                W = W + der_v(i) * (m.transpose() * weights * m);
            }
        }
    }
    return W;
}

// NAME:    dot
// DESC:    computes the dot-product of two TMB vectors
// IN:      v1: vector<Type>
//          v2: vector<Type>
// OUT:     scalar
template<class Type>
Type dot(vector<Type> v1, vector<Type> v2) {
    return v1.cwiseProduct(v2).sum();
}

// NAME: expand_Q
// DESC: Takes an array of independent sigmas and creates a sparse
//       precision matrix
// IN:   sigma_vector: vector<Type>
//       v1: vector<int>
//       v2: vector<int>
// OUT:  SparseMatrix<Type>
template<class Type>
SparseMatrix<Type> expand_Q(vector<Type> sigma_vector, vector<int> v1, vector<int> v2) {
    int N = v1.size();
    int S = sigma_vector.size();
    int N_1 = 1;
    int N_v1 = v1.maxCoeff() + 1;
    int N_v2 = v2.maxCoeff() + 1;
    SparseMatrix<Type> Q(N,N);
    for (int n = 0; n < N; n++) {
        if (S == N_1){
            Q.insert(n,n) = 1. / pow(sigma_vector[0], 2.);
        }
        else if (S == N_v1){
            Q.insert(n,n) = 1. / pow(sigma_vector[v1[n]], 2.);
        }
        else if (S == N_v2){
            Q.insert(n,n) = 1. / pow(sigma_vector[v2[n]], 2.);
        }
        else if (S == N){
            Q.insert(n,n) = 1. / pow(sigma_vector[n], 2.);
        }
        else{
            Q.insert(n,n) = 1./1.;
        }
    }
    return Q;
}

// NAME: eval_ar
// DESC: evaluate ar re terms
// IN:   log_sigma_loc: vector<Type>
//       reg: vector<int>
//       sreg: vector<int>
//       log_sigma_age: vector<Type>
//       age: vector<int>
//       logit_rho: Type
// OUT:  Type
template<class Type>
Type eval_ar(vector<Type> log_sigma_loc, vector<int> reg, vector<int> sreg,
             vector<Type> log_sigma_age, vector<int> age, Type logit_rho,
             array<Type> pi){
    SparseMatrix<Type> Q_age;
    SparseMatrix<Type> Q_loc;
    if(log_sigma_age.size() == 0){
        Q_age = expand_Q(log_sigma_age, age, age);
    }
    else{
        Q_age = expand_Q(exp(log_sigma_age), age, age);
    }
    if(log_sigma_loc.size() == 0){
        Q_loc = expand_Q(log_sigma_loc, reg, sreg);
    }
    else{
        Q_loc = expand_Q(exp(log_sigma_loc), reg, sreg);
    }
    Type rho = expit(logit_rho);
    Type nll = SEPARABLE(SCALE(AR1(rho), pow(1. / (1. -pow(rho, 2.)), .5)),
                         SEPARABLE(GMRF(Q_age), GMRF(Q_loc)))(pi);
    return nll;
}

// NAME: empty_replace
// DESC: replaces an empty vector with a vector of zeros with proper pred length
// IN:   param_array: array<Type>
//       zero_array: array<Type>
// OUT:  array<Type>
template<class Type>
array<Type> empty_replace(array<Type> param_array, array<Type> zero_array){
    if(param_array.dim(0) > 0){
        return(param_array);
    }
    return(zero_array);
}

// OBJECTIVE FUNCTION
template<class Type>
Type objective_function<Type>::operator() () {

    // printf ("%s \n", "Testing the print function inside of TMB C++");

// DATA
    // OUTCOME VARIABLE
        DATA_ARRAY(y_U); // underlying ln(rate) [LxAxT]
        DATA_ARRAY(y_T); // total ln(rate) [LxAxT]

    // COVARIATES
        DATA_ARRAY(X_age);                              // Age specific covariates
        DATA_ARRAY(X_location);                         // Location specific covariates
        DATA_ARRAY(X_region);                           // Region specific covariates
        DATA_ARRAY(X_super_region);                     // Super Region specific covariates
        DATA_ARRAY(X_region_age);                       // Region Age specific covariates
        DATA_ARRAY(X_super_region_age);                 // Super Region Age specific covariates
        DATA_ARRAY(X_location_age);                     // Location Age covariates
        DATA_ARRAY(X_global);                           // Global covariates
        DATA_ARRAY(X2);                                 // Second level covariates by age
        DATA_ARRAY(Z_age);                              // Age specific random effects covariates
        DATA_ARRAY(Z_location);                         // Location specific random effects covariates
        DATA_ARRAY(Z_region);                           // Region specific random effects covariates
        DATA_ARRAY(Z_super_region);                     // Super region specific random effects covariates
        DATA_ARRAY(Z_region_age);                       // Region age specific random effects covariates
        DATA_ARRAY(Z_super_region_age);                 // Super Region age specific random effects covariates
        DATA_ARRAY(Z_location_age);                     // location age specific random effects covariates
        DATA_ARRAY(Z_global);                           // global random effects covariates
        DATA_ARRAY(constant);                           // consntant to add to model
        DATA_ARRAY(constant_mult);                      // ones array to multiply constant by purely for convenience
        DATA_VECTOR(beta_age_constraint);               // priors on beta [K]
        DATA_VECTOR(beta_location_constraint);          // constraints on fixed effects variables
        DATA_VECTOR(beta_region_constraint);
        DATA_VECTOR(beta_region_age_constraint);
        DATA_VECTOR(beta_super_region_constraint);
        DATA_VECTOR(beta_super_region_age_constraint);
        DATA_VECTOR(beta_location_age_constraint);
        DATA_VECTOR(beta_global_constraint);
        DATA_VECTOR(beta2_constraint);

    // INDICATOR VARIABLES
        DATA_IVECTOR(region);       // length equal to locations indicating which region a location is in
        DATA_IVECTOR(super_region); // length equal to locations indicating which super-region a location is in
        DATA_IVECTOR(has_risks);    // indicator for which age groups have risk factors
        DATA_IVECTOR(age);          // vector of each unique age
        DATA_IVECTOR(location);     // vector of each unique location


    // PRIORS ON OMEGA
        DATA_SCALAR(omega_loc_U);
        DATA_SCALAR(omega_age_U);
        DATA_SCALAR(omega_loc_time_U);
        DATA_SCALAR(omega_age_time_U);
        DATA_SCALAR(omega_loc_T);
        DATA_SCALAR(omega_age_T);
        DATA_SCALAR(omega_loc_time_T);
        DATA_SCALAR(omega_age_time_T);
        DATA_SCALAR(weight_decay);
    // Adjustment factor
        DATA_INTEGER(mean_adjust);       // whether to use adjusted values or raw for response variables
        DATA_INTEGER(testing_prints);    // whether to print messages or not
        DATA_INTEGER(holdout_start);     // what year to start the forecasts
        test_print("Data has been loaded", testing_prints);

        array<Type> zero_array = zero_3d(y_U); // empty array to use in the linear model for convenience

    // printf ("%s \n", "The data was loaded!");

    // STORE SIZES
        size_t L = y_T.dim(0);        // number of locations
        size_t A = y_T.dim(1);        // number of ages
        size_t T = y_T.dim(2);        // number of years
        size_t K2 = X2.dim(3);        // number of covariates at the second level of model


    // printf ("%s \n", "sizes have been assigned!");

    // STORE LOOKUPS FOR WHETHER THERE ARE RISKS
        size_t R = 0; // number of age groups that have risks
        vector<int> A_to_R_map(A); // map each age group to its position in the list of only ages with risks
        for (size_t a = 0; a < A; a++) {
            if (has_risks(a) == 1) {
                A_to_R_map(a) = R;
                R += 1;
            }
            else {
                A_to_R_map(a) = -1;
            }
        }

// PARAMETERS
    // COEFFICIENTS
        PARAMETER_ARRAY(beta_age_raw);     // [AxK]
        PARAMETER_ARRAY(beta_location_raw);     // [LxK]
        PARAMETER_ARRAY(beta_region_raw);     // [RxK]
        PARAMETER_ARRAY(beta_super_region_raw);     // [SRxK]
        PARAMETER_ARRAY(beta_region_age_raw);     // [RXAxK]
        PARAMETER_ARRAY(beta_super_region_age_raw);     // [SRxAxK]
        PARAMETER_ARRAY(beta_location_age_raw);     // [LxAxK]
        PARAMETER_ARRAY(beta_global_raw);     // [K]
        PARAMETER_VECTOR(log_zeta_D);   // coefficient on underlying by age [R]
        PARAMETER_ARRAY(beta2_raw);   // coefficient on risk scalar by age [R,K]

    // RANDOM EFFECTS
        PARAMETER_ARRAY(gamma_location);        // random slope by location [LxA]
        PARAMETER_ARRAY(log_tau_location);
        PARAMETER_ARRAY(gamma_region);        // random slope by region [R]
        PARAMETER_ARRAY(log_tau_region);
        PARAMETER_ARRAY(gamma_super_region);        // random slope by super_region [SR]
        PARAMETER_ARRAY(log_tau_super_region);
        PARAMETER_ARRAY(gamma_location_age);        // random slope by location/age [LxA]
        PARAMETER_ARRAY(log_tau_location_age);
        PARAMETER_ARRAY(gamma_region_age);        // random slope by region-age [RxA]
        PARAMETER_ARRAY(log_tau_region_age);
        PARAMETER_ARRAY(gamma_super_region_age);        // random slope by super_region-age [SRxA]
        PARAMETER_ARRAY(log_tau_super_region_age);
        PARAMETER_ARRAY(gamma_age);        // random slope by age [A]
        PARAMETER_ARRAY(log_tau_age);
        PARAMETER_ARRAY(gamma_global);        // random slope global [LxA]
        PARAMETER_ARRAY(log_tau_global);
        PARAMETER_ARRAY(pi); // temporal component to be optionally added
        PARAMETER_VECTOR(log_age_sigma_pi); // for ar terms that differ by age
        PARAMETER_VECTOR(log_location_sigma_pi); // for AR terms that differ by location
        PARAMETER_VECTOR(logit_rho); // temporal autocorrelation

    // DATA VARIANCE
        PARAMETER_VECTOR(log_sigma_U);  // log of data variance on underlying (vector so it can disappear if no risks)
        PARAMETER(log_sigma_T);         // log of data variance on total

        test_print("Parameters have been loaded", testing_prints);

// MODEL
    // INITIALIZE
        parallel_accumulator<Type> nll(this);

    // TRANSFORMATIONS
        // VARIANCE
        Type sigma_U = 0.;
        if (R > 0) {
            sigma_U = exp(log_sigma_U(0)); // if there are underlying deaths
        }
        Type sigma_T = exp(log_sigma_T);
        // beta transforms
        array<Type> beta_age = transform_beta(beta_age_raw, beta_age_constraint);
        array<Type> beta_location = transform_beta(beta_location_raw, beta_location_constraint);
        array<Type> beta_region = transform_beta(beta_region_raw, beta_region_constraint);
        array<Type> beta_super_region = transform_beta(beta_super_region_raw, beta_super_region_constraint);
        array<Type> beta_location_age = transform_beta(beta_location_age_raw, beta_location_age_constraint);
        array<Type> beta_region_age = transform_beta(beta_region_age_raw, beta_region_age_constraint);
        array<Type> beta_super_region_age = transform_beta(beta_super_region_age_raw, beta_super_region_age_constraint);
        array<Type> beta_global = transform_beta(beta_global_raw, beta_global_constraint);
        test_print("Beta transforms made", testing_prints);
        // tau transforms
        array<Type> tau_age = transform_parameter(log_tau_age, Type(1.));
        array<Type> tau_location = transform_parameter(log_tau_location, Type(1.));
        array<Type> tau_region = transform_parameter(log_tau_region, Type(1.));
        array<Type> tau_super_region = transform_parameter(log_tau_super_region, Type(1.));
        array<Type> tau_location_age = transform_parameter(log_tau_location_age, Type(1.));
        array<Type> tau_region_age = transform_parameter(log_tau_region_age, Type(1.));
        array<Type> tau_super_region_age = transform_parameter(log_tau_super_region_age, Type(1.));
        array<Type> tau_global = transform_parameter(log_tau_global, Type(1.));
        test_print("Tau transforms made", testing_prints);
        // second level transforms
        array<Type> beta2 = transform_beta(beta2_raw, beta2_constraint);
        vector<Type> zeta_D(R);
        if (R > 0) {
            for (size_t r = 0; r < R; r++) {
                zeta_D(r) = exp(log_zeta_D(r));
            }
        }

        test_print("zetas calculated", testing_prints);

        // MEANS
        matrix<Type> y_mean_U(L,A);
        matrix<Type> y_mean_T(L,A);
        Type tmp_num_U;
        Type tmp_denom_U;
        Type tmp_num_T;
        Type tmp_denom_T;
        for (size_t l = 0; l < L; l++) {
            for (size_t a = 0; a < A; a++) {
                tmp_num_U = 0.;
                tmp_denom_U = 0.;
                tmp_num_T = 0.;
                tmp_denom_T = 0.;
                for (size_t t = 0; t < T; t++) {
                    if (!CppAD::isnan(y_U(l,a,t))) {
                        tmp_num_U += y_U(l,a,t);
                        tmp_denom_U += 1.;
                    }
                    if (!CppAD::isnan(y_T(l,a,t))) {
                        tmp_num_T += y_T(l,a,t);
                        tmp_denom_T += 1.;
                    }
                }
                y_mean_U(l,a) = tmp_num_U / tmp_denom_U;
                y_mean_T(l,a) = tmp_num_T / tmp_denom_T;
            }
        }

        test_print("location age means calculated", testing_prints);

    // SMOOTHING WEIGHTS
        // LOCATIONS
            // s^+ - s
            // in other words, take -1*neighbors then put rowsum() on the diagonal
            matrix<Type> W_loc(L,L);
            for (size_t l1 = 0; l1 < L; l1++) {
                int r1 = region(l1);
                int s1 = super_region(l1);
                Type s_plus = Type(0.);
                for (size_t l2 = 0; l2 < L; l2++) {
                    int r2 = region(l2);
                    int s2 = super_region(l2);
                    Type n = 0.;
                    if (r1 == r2) Type n = 2.;
                    else if (s1 == s2) Type n = 1.;
                    s_plus += n;
                    W_loc(l1,l2) = -1 * n;
                }
                W_loc(l1,l1) = s_plus;
            }

            test_print("calculated location smoothing weights", testing_prints);

        // AGES
            // smoothness over ages using derivative prior
            vector<Type> age_deriv_n(3);
            age_deriv_n << 0, 1, 1;
            matrix<Type> W_age = derivative_prior<Type>(A, age_deriv_n);

            test_print("calculated age smoothing weights", testing_prints);

        // TIME FIRST DERIVATIVE MATRIX
            matrix<Type> D_time = deriv<Type>(1, T);

    // PREDICTIONS
        // UNDERLYING
            array<Type> mu_U = mult_model(X_age, beta_age, beta_age_raw.dim(3), zero_array) +
                               mult_model(X_location, beta_location, beta_location_raw.dim(3), zero_array) +
                               mult_model(X_location_age, beta_location_age, beta_location_age_raw.dim(3), zero_array) +
                               mult_model_region(X_region, beta_region, beta_region_raw.dim(3), zero_array, region) +
                               mult_model_region(X_region_age, beta_region_age, beta_region_age_raw.dim(3), zero_array, region) +
                               mult_model_region(X_super_region, beta_super_region, beta_super_region_raw.dim(3), zero_array, super_region) +
                               mult_model_region(X_super_region_age, beta_super_region_age, beta_super_region_age_raw.dim(3), zero_array, super_region) +
                               mult_model(X_global, beta_global, beta_global_raw.dim(3), zero_array) +
                               mult_model(Z_age, gamma_age, gamma_age.dim(3), zero_array) +
                               mult_model(Z_location, gamma_location, gamma_location.dim(3), zero_array) +
                               mult_model_region(Z_region, gamma_region, gamma_region.dim(3), zero_array, region) +
                               mult_model_region(Z_region_age, gamma_region_age, gamma_region_age.dim(3), zero_array, region) +
                               mult_model_region(Z_super_region, gamma_super_region, gamma_super_region.dim(3), zero_array, super_region) +
                               mult_model_region(Z_super_region_age, gamma_super_region_age, gamma_super_region_age.dim(3), zero_array, super_region) +
                               mult_model(Z_location_age, gamma_location_age, gamma_location_age.dim(3), zero_array) +
                               mult_model(Z_global, gamma_global, gamma_global.dim(3), zero_array) +
                               mult_model(constant, constant_mult, constant.dim(3), zero_array);

            test_print("Underlying predictions made", testing_prints);
            array<Type> yhat_U = adjust_preds(mu_U, y_mean_U, 0, mean_adjust);
            if (mean_adjust == 0){
                array<Type> mu_U = adjust_preds(yhat_U, y_mean_U, 1, 1);
            }

            yhat_U += empty_replace(pi, zero_array); // add in the optional temporal random effects

            test_print("Underlying predictions adjusted", testing_prints);
            REPORT(mu_U);
            REPORT(yhat_U);

        // TOTAL
            array<Type> yhat_T(L,A,T);
            Type betaX;
            for (size_t l = 0; l < L; l++) {
                for (size_t a = 0; a < A; a++) {
                    for (size_t t = 0; t < T; t++) {
                        if (has_risks(a) == 1) {
                            size_t r = A_to_R_map(a);
                            betaX = 0.;
                            for (size_t k = 0; k < K2; k++)
                                betaX += beta2(0,r,0,k) * X2(l,a,t,k);
                            yhat_T(l,a,t) = (zeta_D(r) * yhat_U(l,a,t)) + (betaX);
                        }
                        else {
                            yhat_T(l,a,t) = yhat_U(l,a,t);
                        }
                    }
                }
            }

            test_print("Total predictions made", testing_prints);
            array<Type> mu_T = adjust_preds(yhat_T, y_mean_T, 1, 1);
            test_print("Total predictions adjusted", testing_prints);
            REPORT(mu_T);
            REPORT(yhat_T);

    // FIRST DERIVATIVES OF MU OVER TIME
        CppAD::vector< vector<Type> > first_deriv_U(L*A);
        CppAD::vector< vector<Type> > first_deriv_T(L*A);
        vector<Type> mu_tmp_U(T);
        vector<Type> mu_tmp_T(T);
        CppAD::vector< vector<Type> > mu_vec_U(L*A);
        CppAD::vector< vector<Type> > mu_vec_T(L*A);
        for (size_t l = 0; l < L; l++) {
            for (size_t a = 0; a < A; a++) {
                for (size_t t = 0; t < T; t++) {
                    mu_tmp_U(t) = mu_U(l,a,t);
                    mu_tmp_T(t) = mu_T(l,a,t);
                }
                mu_vec_U[l*A + a] = mu_tmp_U;
                mu_vec_T[l*A + a] = mu_tmp_T;
                first_deriv_U[l*A + a] = D_time * mu_tmp_U;
                first_deriv_T[l*A + a] = D_time * mu_tmp_T;
            }
        }

    // PRIORS
        // AGE SMOOTHING
            for (size_t l = 0; l < L; l++) {
                for (size_t a1 = 0; a1 < A; a1++) {
                    for (size_t a2 = 0; a2 <= a1; a2++) {
                        if (W_age(a1,a2) != 0.) {
                            if ((has_risks(a1) == 1) & (has_risks(a2) == 1) & (omega_age_U > 0)) {
                                nll -= replace_nan((-1./2.) * (omega_age_U / (L*T)) * W_age(a1,a2) * (a1 != a2 ? 2. : 1.) *
                                    dot(mu_vec_U[l*A + a1], mu_vec_U[l*A + a2]));
                            }
                            if(omega_age_T > 0){
                                nll -= replace_nan((-1./2.) * (omega_age_T / (L*T)) * W_age(a1,a2) * (a1 != a2 ? 2. : 1.) *
                                    dot(mu_vec_T[l*A + a1], mu_vec_T[l*A + a2]));
                            }
                        }
                    }
                }
            }

        // LOCATION SMOOTHING
            for (size_t l1 = 0; l1 < L; l1++) {
                for (size_t l2 = 0; l2 <= l1; l2++) {
                    for (size_t a = 0; a <= A; a++) {
                        if (W_loc(l1,l2) != 0.) {
                            if ((has_risks(a) == 1) & (omega_loc_U > 0)) {
                                nll -= replace_nan((-1./2.) * (omega_loc_U / (L*T)) * W_loc(l1,l2) * (l1 != l2 ? 2. : 1.) *
                                dot(mu_vec_U[l1*A + a], mu_vec_U[l2*A + a]));
                            }
                            if(omega_loc_T > 0){
                                nll -= replace_nan((-1./2.) * (omega_loc_T / (L*T)) * W_loc(l1,l2) * (l1 != l2 ? 2. : 1.) *
                                    dot(mu_vec_T[l1*A + a], mu_vec_T[l2*A + a]));
                            }
                        }
                    }
                }
            }

        // AGE-TIME SMOOTHING
            for (size_t l = 0; l < L; l++) {
                for (size_t a1 = 0; a1 < A; a1++) {
                    for (size_t a2 = 0; a2 <= a1; a2++) {
                        if (W_age(a1,a2) != 0.) {
                            if ((has_risks(a1) == 1) & (has_risks(a2) == 1) & (omega_age_time_U > 0)) {
                                nll -= replace_nan((-1./2.) * (omega_age_time_U / (L*T)) * W_age(a1,a2) * (a1 != a2 ? 2. : 1.) *
                                dot(first_deriv_U[l*A + a1], first_deriv_U[l*A + a2]));
                            }
                            if(omega_age_time_T > 0){
                                nll -= replace_nan((-1./2.) * (omega_age_time_T / (L*T)) * W_age(a1,a2) * (a1 != a2 ? 2. : 1.) *
                                    dot(first_deriv_T[l*A + a1], first_deriv_T[l*A + a2]));
                            }
                        }
                    }
                }
            }

        // LOCATION-TIME SMOOTHING
            for (size_t l1 = 0; l1 < L; l1++) {
                for (size_t l2 = 0; l2 <= l1; l2++) {
                    for (size_t a = 0; a <= A; a++) {
                        if (W_loc(l1,l2) != 0.) {
                            if ((has_risks(a) == 1) & (omega_loc_time_U > 0)) {
                                nll -= replace_nan((-1./2.) * (omega_loc_time_U / (L*T)) * W_loc(l1,l2) * (l1 != l2 ? 2. : 1.) *
                                dot(first_deriv_U[l1*A + a], first_deriv_U[l2*A + a]));
                            }
                            if (omega_loc_time_T > 0){
                                nll -= replace_nan((-1./2.) * (omega_loc_time_T / (L*T)) * W_loc(l1,l2) * (l1 != l2 ? 2. : 1.) *
                                    dot(first_deriv_T[l1*A + a], first_deriv_T[l2*A + a]));
                            }
                        }
                    }
                }
            }

            test_print("GK priors applied", testing_prints);

        // THETA SECULAR TRENDS
            nll -= re_prior(gamma_location_age, tau_location_age);
            nll -= re_prior(gamma_age, tau_age);
            nll -= re_prior(gamma_location, tau_location);
            nll -= re_prior(gamma_region, tau_region);
            nll -= re_prior(gamma_super_region, tau_super_region);
            nll -= re_prior(gamma_region_age, tau_region_age);
            nll -= re_prior(gamma_super_region_age, tau_super_region_age);
            nll -= re_prior(gamma_global, tau_global);
            if (logit_rho.size() > 0){
                nll += eval_ar(log_location_sigma_pi, region, super_region,
                               log_age_sigma_pi, age, logit_rho[0], pi);// gotta plus here because of namespace density
            }

            test_print("Random effect priors applied", testing_prints);

    // DATA LIKELIHOOD
        size_t last_insample = 0;
        for (size_t t = 0; t < T; t++) {
            if (!CppAD::isnan(y_T(0,0,t))) last_insample = t;
        }
        for (size_t l = 0; l < L; l++) {
            for (size_t a = 0; a < A; a++) {
                for (size_t t = 0; t <= last_insample; t++) {
                    if (!CppAD::isnan(yhat_U(l,a,t)) & !CppAD::isnan(y_U(l,a,t)) & (has_risks(a) == 1) & (t < holdout_start))
                        nll -= dnorm(y_U(l,a,t), yhat_U(l,a,t),
                                     sigma_U * Type(pow(Type(last_insample - t + 1), weight_decay)), true);
                    if (!CppAD::isnan(yhat_T(l,a,t)) & !CppAD::isnan(y_T(l,a,t))  & (t < holdout_start))
                        nll -= dnorm(y_T(l,a,t), yhat_T(l,a,t),
                                     sigma_T * Type(pow(Type(last_insample - t + 1), weight_decay)), true);
                }
            }
        }

        if (testing_prints == 1){
            REPORT(beta_age);
            REPORT(sigma_T);
            REPORT(gamma_location_age);
            REPORT(log_tau_location_age);
            REPORT(yhat_T);
            REPORT(yhat_U);
            REPORT(y_mean_U);
            REPORT(y_mean_T);
        }

    // FIN
        return nll;
}
