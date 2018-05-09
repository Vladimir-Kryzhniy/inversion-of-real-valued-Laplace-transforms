#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''''
    Implementation and examples of:
     V V Kryzhniy  On regularization method for numerical inversion of the Laplace transforms computable at any point on the real axis
                   Journal of Inverse and Ill-Posed Problems,18(4), 2010
    Examples demonstrate limitations inherent in inversion of real-valued Laplace transforms:
    Kryzhniy V. V. Numerical inversion of the Laplace transform: Analysis via regularized analytic continuation. 
                   Inverse Problems 22, 2006
    
    
    HTF - H. Bateman, A. Erdelyi Higher Transcendental Functions, Volume 2
    NR  W.H.Press at al, Numerical Recipes, any edition 

    @author: Vladimir Kryzhniy, April 2018
'''

import numpy as np
from scipy.special import gamma
from numpy import linalg 
from scipy.optimize import minimize
from de_quadrature import intdeo

class InvertLT(object):
    ''' Invert a Laplace transforms computable at any point on the real axis.
        Input parameters:
            image - a function that comtutes a Laplace transform at any p > 0
            t - a numpy array of points to compute inverse Laplace transform; t > 0
        Optional parameters (recommended): 
            digits - input precision; the number of correct digits in Laplace transform 
            pw - a power of asymptotic of F(p) ~ 1 / p**pw as p -> 0
            pw = np.inf when p**pw / F(p) -> 0 for any pw 
        params = (a, alpha, r)
        
        
        1. Invert F1 using information input precision and asymptotic of F(p) as p -> 0
            ret= iltinvert(F1,t_array,digits1, pw1)
            inverse = ret[0]
        2. Invert F1 using known parameters a, alpha, r:
            ilt = InvertLT()
            ret= ilt.invert(F1,t_array,params = (a,alpha,r))
            inverse = ret[0]
        3. No additional information is known. Program attepmts to compute appropriate papameters 
           by solving a minimization problem and using a default value as a starting point.
           ret= ilt.invert(F1,t_array)
           inverse = ret[0]
    '''

    def __init__(self):
        pass
    def _set_params(self, image, tarray, digits, pw, params):
        '''set parameters for inverting another Laplace transform'''
        assert(all(tarray > 0))
        self.tarray = np.sort(tarray)
        self.imagefunc = image
        #storages for reusing computed values
        self.lt = {}
        self.krn1 = {}
        if params !=  None:
            self.a, self.alpha, self.r = params
            return
        self.krn2 = {}
        self.r = digits * 4 / 3
        if np.isinf(pw):
            self.a = 1
            self.alpha = 1 
        else:
            self.a = -abs(pw) 
            self.alpha = 0
            self.t = self.tarray[0]
            res = self._discrepancy((self.a,self.alpha,self.r))
            if abs(res) > 1e6: # bad initialization, use default parameters
                self.a = 1
                self.alpha = 1      

    def invert(self, image, tarray, digits = 15, pw = np.inf, params = None):
        ''' Computes the inverse Laplace transform for given input.
            Returns the computed inverse (ret[0]) and parameters (ret[1])
        ''' 
        self._set_params(image, tarray, digits, pw, params)
        if params ==  None:  
            res = minimize(self._discrepancy,(self.a, self.alpha, self.r), method = 'powell',
                           options = {'xtol': 0.1,'ftol': 0.1})#, 'maxfev': 100})
            self.a, self.alpha, self.r = res.x
            self.alpha = max(self.alpha,0)
            res.x = (self.a, self.alpha, self.r)
            print(res)
        self.krn1 = {}
        return self._ilt1(), (self.a, self.alpha, self.r)
     
    def _discrepancy(self, params):
        self.a, self.alpha, self.r = params
        self.alpha = max(self.alpha,0)
        self.gammainv = 1 / gamma(-self.a -1j*self.r)
        self.gammainv2 = 1 / gamma(-self.a -0.5 -1j*self.r)
        self.krn1 = {}
        self.krn2 = {}
        fctr = np.exp(self.alpha)/np.pi
        diff = np.zeros(len(self.tarray))
        eps = max(10**(-self.r - 0.5), 1.e-15)
        for k in np.arange(0,len(self.tarray)):
            self.t = self.tarray[k]
            res, err = intdeo(self._integrand_diff,0,self.r,eps)
            diff[k] = res * fctr/self.t
        ret =  linalg.norm(diff)
#        print('a =', self.a, 'alpha = ', self.alpha, 'R =', self.r, 'dscr=', ret)
        return ret

    def _integrand_diff(self,x):
        u = (np.exp(x)+self.alpha)/self.t
        v = (np.exp(-x)+self.alpha)/self.t
        if u in self.lt.keys():  
            lt_u, lt_v = self.lt[u]
        else:
            lt_u = self.imagefunc(u)
            lt_v = self.imagefunc(v)
            self.lt[u] = (lt_u, lt_v)
        if x in self.krn1.keys():
            kp, km = self.krn1[x]
            kp2, km2= self.krn2[x]           
        else:
            kp = self._kernel1(x)
            km = self._kernel1(-x)
            kp2 = self._kernel2(x)
            km2 = self._kernel2(-x)
            self.krn1[x] = (kp, km)
            self.krn2[x] = (kp2, km2)
        return lt_u*(kp - kp2) + lt_v*(km - km2)
       
    def _ilt1(self):
        self.gammainv = 1 / gamma(-self.a -1j*self.r)
        fctr = np.exp(self.alpha)/np.pi
        ilt = np.zeros(len(self.tarray))
        eps = max(10**(-self.r - 0.5), 1.e-15)
        for k in np.arange(0,len(self.tarray)):
            self.t = self.tarray[k]
            res, err = intdeo(self._integrand1,0,self.r,eps)
            ilt[k] = res * fctr/self.t
        return ilt

    def _integrand1(self,x):
        u = (np.exp(x)+self.alpha)/self.t
        v = (np.exp(-x)+self.alpha)/self.t
        if u in self.lt.keys():  
            lt_u, lt_v = self.lt[u]
        else:
            lt_u = self.imagefunc(u)
            lt_v = self.imagefunc(v)
            self.lt[u] = (lt_u, lt_v)
        if x in self.krn1.keys():
            kp, km = self.krn1[x]
        else:
            kp = self._kernel1(x)
            km = self._kernel1(-x)
            self.krn1[x] = (kp, km)
        return lt_u*kp + lt_v*km

    def _kernel1(self,x):
        y = np.exp(x) 
        if(y >= self.r/3):
            kernel = -self._psi(-self.a - 1j*self.r,y)			
        else:
            kernel = self._phi1(-self.a -1j*self.r, y)
        kernel *= self.gammainv
        kernel *= np.exp((1 - self.a) * x -1j * self.r * x)
        return kernel.imag

    def _ilt2(self):
        self.gammainv2 = 1 / gamma(-self.a -0.5 -1j*self.r)
        fctr = np.exp(self.alpha)/np.pi
        ilt = np.zeros(len(self.tarray))
        eps = max(10**(-self.r - 0.5), 1.e-15)
        for k in np.arange(0,len(self.tarray)):
            self.t = self.tarray[k]
            res, err = intdeo(self._integrand2,0,self.r,eps)
            ilt[k] = res * fctr/self.t
        return ilt 

    def _integrand2(self,x):
        u = (np.exp(x)+self.alpha)/self.t
        v = (np.exp(-x)+self.alpha)/self.t
        if u in self.lt.keys():  
            lt_u, lt_v = self.lt[u]
        else:
            lt_u = self.imagefunc(u)
            lt_v = self.imagefunc(v)
            self.lt[u] = (lt_u, lt_v)
        if x in self.krn2.keys():
            kp, km = self.krn2[x]
        else:
            kp = self._kernel2(x)
            km = self._kernel2(-x)
            self.krn2[x] = (kp, km)
        return lt_u*kp + lt_v*km
    
    def _kernel2(self,x):
        y = np.exp(x)
        if(y >= self.r/3):
            kernel2 = -self._psi(-self.a - 0.5 - 1j*self.r,y)			
        else:
            kernel2 = self._phi1(-self.a - 0.5 -1j*self.r, y)
        kernel2 -= self._phi2(self.a - 0.5 -1j*self.r, y)
        kernel2 *= self.gammainv2
        kernel2 *= np.exp((0.5 - self.a) * x -1j * self.r * x)
        return kernel2.imag
    
    def _phi1(self,a, x):
        ''' Computing  sum in (NR: 6.2.5)'''
        res0 = term = res = 1 / a
        ap = a
        while 1:
            ap +=  1
            term *= x / ap
            res0 +=  term
            if res == res0:
                break
            res =  res0
        return res
    def _psi(self,a,x):
        ''' Computing continuous fraction (NR: 6.2.7) '''
        tiny = 1e-30
        b = x + 1 - a
        c = 1 / tiny 
        d = h = 1 / b
        i = 0
        while 1:
            i += 1
            an = i*(a - i)
            b += 2
            d = an*d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            dl = d * c
            h *= dl
            if h == h*dl:
                break
        return h
    
    def _phi2(self,a,x):
        '''  Computing sum (HTF: 9.2.4_2) and asymptotic expansion (HTF: 9.2.6) '''
        if abs(x) > 5*abs(a+20):
            res0 = res = fct = 1/x
            for k in np.arange(1,19):
                fct *= (k - a)/x
                res0 += fct
                if res == res0:
                    break
                res = res0
            return res
        res0 = res = 1 / a
        k = fct = 1
        while 1:
            fct *= x / k
            res0 += fct/(a + k)
            if res == res0:
                break
            res = res0
            k += 1
        res *= np.exp(-x)
        return res
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.special
    ilt = InvertLT()
    def F1(x):
        return np.log(x)/x
    def f1(t):
            return -np.log(t) - 0.57721566490153
    t = np.linspace(0.01,100,endpoint=True)
    ret = ilt.invert(F1,t,digits = 15, pw = 1)
    ilt_num = ret[0]
    ilt_exact = f1(t)
    plt.plot(t,ilt_exact,'b',label = 'ilt_exact')
    plt.plot(t,ilt_num,'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_1(t)$')
    plt.title(r'$f_1(t) = log(t) - \gamma$')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F2(x):
        return np.exp(1/x)/x
    def f2(t):
        return scipy.special.iv(0,2*np.sqrt(t))
    t = np.linspace(0.01,15)
    ret = ilt.invert(F2,t,15,np.inf)
    plt.plot(t,f2(t),'b',label = 'ilt_exact')
    plt.plot(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_2(t)$')
    plt.title(r'$f_2(t) = I_0(2\sqrt{t})$')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F3(x):
        return  256/np.power((4+x),4)
    def f3(t):
        return 128*np.power(t,3)*np.exp(-4*t)/3
    t = np.logspace(-2,1,num = 100)
    ret = ilt.invert(F3,t,15,0)
    plt.semilogx(t,f3(t),'b',label = 'ilt_exact')
    plt.semilogx(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_3(t)$')
    plt.title(r'$f_3(t) = \frac{128}{3}t^{3}\mathrm{e}^{-4t}$')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F4(x):
        return np.exp(-np.sqrt(x/2))
    def f4(t):
        return np.exp(-0.125/t)/(2*t*np.sqrt(2*np.pi*t))
    t = np.logspace(-2,1,num = 100)
    ret = ilt.invert(F4,t,15,0)
    plt.semilogx(t,f4(t),'b',label = 'ilt_exact')
    plt.semilogx(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_4(t)$')
    plt.title(r'$f_4(t) = \frac{exp(-1/8t)}{2t\sqrt{2\pi t}}$')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F4a(x):
        return F3(x) + F4(x)
    def f4a(t):
        return f3(t) + f4(t)
    t = np.logspace(-2,1,num = 100)
    ret = ilt.invert(F4a,t,15,0)
    plt.semilogx(t,f4a(t),'b',label = 'ilt_exact')
    plt.semilogx(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_{4a}(t)$')
    plt.title('Example 4a')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F5(x):
        '''
            Finite differences method for
            Y''(x) -pY=0; Y(0)=0; Y'(1)=1; 
            n = 32 knots
            output Y(0.5)
        '''
        if x == 0:
            return  x
        n = 32
        h = 1.0 / n
        a = np.zeros(n + 2)
        b = np.zeros(n + 2)
        c = np.zeros(n + 2)
        d = np.zeros(n + 2)
        ksi = np.zeros(n + 2)
        eta = np.zeros(n + 2)
        y = np.zeros(n + 2)
        b[0] = 1.0
        for i in range(1, n+1, 1):
            a[i] = 1.0
            b[i] = 2 + x * h * h
            c[i] = 1.0
            d[i] = 0.0
            b[n] /=  2
            c[n] = 0.0
            d[n] = -h
            ksi[0] = 0.0
            eta[0] = 0.0
        for i in range(0, n + 1):
            zn = b[i] - a[i] * ksi[i]
            ksi[i + 1] = c[i] / zn
            eta[i + 1] = (a[i] * eta[i] - d[i]) / zn
            y[n] = 0.0 
        for i in range(n, -1, -1):
            y[i] = ksi[i + 1] * y[i +1 ] + eta[i + 1]
        return y[16] / x

    def f5(t):
        x = 0.5
        res = x
        ad = 1.0
        n = 0 
        fct = -2
        while 1:
            d = (n + 0.5) * np.pi
            ad = fct * np.exp(-d * d * t) / (d * d)
            res += ad * np.sin(d * x)
            n += 1
            fct *= -1
            if max(abs(ad)) < 1.0E-12:
                break
        return res
    t = np.linspace(0.01,2); 
    ret = ilt.invert(F5,t,7,1)
    plt.plot(t,f5(t),'b',label = 'ilt_exact')
    plt.plot(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_5(t)$')
    plt.title(r'Example 5')
    plt.grid(True)
    plt.legend()
    plt.show()

    #Oscillating function. Can be inverted accurately only for relatively small $t$
    def F6(x):
        return np.exp(-x/np.sqrt(x*x+1))/x
    def f6(t):
        '''see Duffy, Dean G. Transform methods for solving partial differential equations. 
            Chapman & Hall/CRC, 2004
        '''
        nn = 1000
        x = 1.0
        b = 1.0
        sm = 0
        eta = 0
        d_eta = b / nn
        for n in range(0, nn -1, 2):
            if n == 0:
                sm = np.cos(eta * t) * x / b
            else:
                sm += np.cos(eta * t) * np.sin( x* eta / np.sqrt(b * b - eta * eta)) / eta
            eta += d_eta
            sm += 4 * np.cos(eta * t) * np.sin( x * eta / np.sqrt(b * b - eta * eta)) / eta
            eta += d_eta
            if eta < b:
                sm += np.cos(eta * t) * np.sin(x * eta / np.sqrt(b * b - eta * eta)) / eta
        return  1 - 2 * d_eta * sm.real / (3 * np.pi)
    t = np.linspace(0.1,20,num = 100)
    ret = ilt.invert(F6,t,15,1)
    plt.plot(t,f6(t),'b',label = 'ilt_exact')
    plt.plot(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_5(t)$')
    plt.title(r'Example 6')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F7(x):
        return 1/np.sqrt(x*x+1)
    def f7(t):
        return scipy.special.jv(0,t)
    t = np.linspace(0.01,20)
    ret = ilt.invert(F7,t,15,0)
    plt.plot(t,f7(t),'b',label = 'ilt_exact')
    plt.plot(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_7(t)$')
    plt.title(r'Example 7')
    plt.grid(True)
    plt.legend()
    plt.show()

    #sum of delta functions
    peaks = [(1,0.5), (2,0.8), (4,2), (100,20)]
    def F8(x):
        ret = 0
        for ampl,t0 in peaks:
            ret += ampl*np.exp(-t0*x)
        return ret
    def sum_delta_r(t):
        def delta_r(tau):
            global a, alpha, r
            dlt_r = np.exp(-alpha*tau/t)*np.sin(r*np.log(tau/t))*np.power(tau/t, a )/(tau - t)
            return dlt_r*np.exp(alpha)/np.pi
        ret = np.zeros_like(t)
        for ampl,t0 in peaks:
            ret += ampl*delta_r(t0)
        return ret
    t = np.logspace(-1,2, num = 500)
    ret = ilt.invert(F8,t,params = (- 0.5, 0, 20))
    a = ret[1][0]
    alpha = ret[1][1]
    r = ret[1][2]
    plt.semilogx(t,sum_delta_r(t),'b',label = 'ilt_theoretical')
    plt.semilogx(t,ret[0],'r--',label='ilt_num')
    plt.xlabel('$t$')
    plt.ylabel('inverse')
    plt.title('')
    plt.grid(True)
    plt.legend()
    plt.show()

    #pulse function
    def F9(x):
        return (np.exp(-x) - np.exp(-2*x))/x
    def f9(t):
        return (t >= 1)*(t <= 2)
    t = np.logspace(-1,1, num = 100)
    ret = ilt.invert(F9,t,15,1)
    plt.semilogx(t,f9(t),'b',label = 'ilt_exact')
    plt.semilogx(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_=9(t)$')
    plt.title('Example 9')
    plt.grid(True)
    plt.legend()
    plt.show()

    def F10(x):
        return x**2/((x**2 +0.25)**3)
    def f10(t):
        t1 = t/2
        return (1 +t1**2)*np.sin(t1) - t1*np.cos(t1)
    t = np.linspace(0.01,30,endpoint=True)
    ret = ilt.invert(F10,t,15,1)
    plt.plot(t,f10(t),'b',label = 'ilt_exact')
    plt.plot(t,ret[0],'r--',label='ilt_num')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_{10}(t)$')
    plt.title('Example 10')
    plt.grid(True)
    plt.legend()
    plt.show()
