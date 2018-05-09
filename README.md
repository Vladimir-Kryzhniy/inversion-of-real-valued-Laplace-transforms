## inversion-of-real-valued-Laplace-transforms

Implementation of inversion of real valued Laplace transforms [1,2].

See InverLT.ipynb for more details

References:
1. Kryzhniy V. V. On regularization method for numerical inversion of the Laplace transforms computable at any point on the real axis. Journal of Inverse and Ill-Posed Problems,18(4), 2010
2. Kryzhniy V. V. Numerical inversion of the Laplace transform: Analysis via regularized analytic continuation. Inverse Problems 22, 2006
3. H. Bateman, A. Erdelyi, Higher Transcendental Functions, Volume 2
4. H. Bateman, A. Erdelyi, Tables of Integral Transforms, Volume 1
5. W.H.Press at al, Numerical Recipes, any edition

''' Invert a Laplace transforms computable at any point on the real axis.
        Input parameters:
            image - a function that comtutes a Laplace transform at any p > 0;
            t - a numpy array of points to compute inverse Laplace transform, t > 0;
        Optional parameters (recommended): 
            digits - input precision; the number of correct digits in Laplace transform 
            pw - a power of asymptotic of F(p) ~ 1 / p^pw as p -> 0
            pw = np.inf when p^pw / F(p) -> 0 for any pw; 
        Method's free parameters (a, alpha, r)
        1. Invert a Laplace transform F1 using information input precision and asymptotic of F(p) as p -> 0
            ret= iltinvert(F1,t_array,digits1, pw1)
            inverse = ret[0]
        2. Invert a Laplace transform F1 using known parameters a, alpha, r:
            ilt = InvertLT()
            ret= ilt.invert(F1,t_array,params = (a,alpha,r))
            inverse = ret[0]
        3. No additional information is known. Program attepmts to compute appropriate papameters 
           by solving a minimization problem and using a default value as a starting point.
           ret= ilt.invert(F1,t_array)
           inverse = ret[0]
    '''
    
