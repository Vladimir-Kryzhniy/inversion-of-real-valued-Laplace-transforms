#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:00:25 2017
Porting
Ooura's implementation of DE-Quadrature 
http://www.kurims.kyoto-u.ac.jp/~ooura/intde.html

@author: Vladimir Kryzhniy
"""
import math

def intde(f, a, b, eps):
    """
        intde        I = integral of f(x) over (a,b)
        f         : integrand f(x)
        a         : lower limit of integration (double)
        b         : upper limit of integration (double)
        eps       : relative error requested (double)
        i         : approximation to the integral
        err       : estimate of the absolute error 
        function  f(x) needs to be analytic over (a,b).
        relative error
            eps is relative error requested excluding
            cancellation of significant digits.
            i.e. eps means : (absolute error) /
                             (integral_a^b |f(x)| dx).
            eps does not mean : (absolute error) / I.
        error message
            err >= 0 : normal termination.
            err < 0  : abnormal termination (m >= mmax).
                       i.e. convergent error is detected :
                           1. f(x) or (d/dx)^n f(x) has
                              discontinuous points or sharp
                              peaks over (a,b).
                              you must divide the interval
                              (a,b) at this points.
                           2. relative error of f(x) is
                              greater than eps.
                           3. f(x) has oscillatory factor
                              and frequency of the oscillation
                              is very high.
    """
#     ---- adjustable parameter ----
    mmax = 256
    efs = 0.1
    hoff = 8.5
#    int m;
#    double pi2, epsln, epsh, h0, ehp, ehm, epst, ba, ir, h, iback,
#        irback, t, ep, em, xw, xa, wg, fa, fb, errt, errh, errd;
    pi2 = math.pi/2
    epsln = 1 - math.log(efs * eps)
    epsh = math.sqrt(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1 / ehp
    epst = math.exp(-ehm * epsln)
    ba = b - a
    ir = f((a + b) * 0.5) * (ba * 0.25)
    i = ir * math.pi
    err = abs(i) * epst
    h = 2 * h0
    m = 1
    while 1:
        iback = i
        irback = ir
        t = h * 0.5
        while 1:
            em = math.exp(t)
            ep = pi2 * em
            em = pi2 / em
            while 1:
                xw = 1. / (1 + math.exp(ep - em))
                xa = ba * xw
                wg = xa * (1 - xw)
                fa = f(a + xa) * wg
                fb = f(b - xa) * wg
                ir += fa + fb
                i += (fa + fb) * (ep + em)
                errt = (abs(fa) + abs(fb)) * (ep + em)
                if m == 1:
                    err += errt * epst
                ep *= ehp
                em *= ehm
                if not ((errt > err) | (xw > epsh)):
                    break                 
            t += h
            if not t < h0:
                break 
        if m == 1:
            errh = (err / epst) * epsh * h0
            errd = 1 + 2 * errh
        else:
            errd = h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))                
        h *= 0.5
        m *= 2
        if not ((errd > errh) & (m < mmax)):
            break
    i *= h
    if (errd > errh):
        err = -errd * m
    else:
        err = errh * epsh * m / (2 * efs)
    return i, err


def intdei(f, a, eps):
    """
    intdei 
    I = integral of f(x) over (a,infinity),
    f(x) has not oscillatory factor.
        f         : integrand f(x)
        a         : lower limit of integration (double)
        eps       : relative error requested (double)
        i         : approximation to the integral
        err       : estimate of the absolute error
        function
            f(x) needs to be analytic over (a,infinity).
        relative error
            eps is relative error requested excluding
            cancellation of significant digits.
            i.e. eps means : (absolute error) /
                             (integral_a^infinity |f(x)| dx).
            eps does not mean : (absolute error) / I.
        error message
            err >= 0 : normal termination.
            err < 0  : abnormal termination (m >= mmax).
                       i.e. convergent error is detected :
                           1. f(x) or (d/dx)^n f(x) has
                              discontinuous points or sharp
                              peaks over (a,infinity).
                              you must divide the interval
                              (a,infinity) at this points.
                           2. relative error of f(x) is
                              greater than eps.
                           3. f(x) has oscillatory factor
                              and decay of f(x) is very slow
                             as x -> infinity                             
    """
#     ---- adjustable parameter ----
    mmax = 256
    efs = 0.1; hoff = 11.0
#    int m;
#    double pi4, epsln, epsh, h0, ehp, ehm, epst, ir, h, iback, irback,
#        t, ep, em, xp, xm, fp, fm, errt, errh, errd;
    pi4 = math.pi/4
    epsln = 1 - math.log(efs * eps)
    epsh = math.sqrt(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1 / ehp
    epst = math.exp(-ehm * epsln)
    ir = f(a + 1)
    i = ir * (2 * pi4)
    err = abs(i) * epst
    h = 2 * h0
    m = 1
    while 1:
        iback = i
        irback = ir
        t = h * 0.5
        while 1:
            em = math.exp(t)
            ep = pi4 * em
            em = pi4 / em
            while 1:
                xp = math.exp(ep - em)
                xm = 1 / xp
                fp = f(a + xp) * xp
                fm = f(a + xm) * xm
                ir += fp + fm
                i += (fp + fm) * (ep + em)
                errt = (abs(fp) + abs(fm)) * (ep + em)
                if m == 1:
                    err += errt * epst 
                ep *= ehp
                em *= ehm
                if not ((errt > err) | (xm > epsh)):
                    break
            t += h
            if not t < h0:
                break
        if m == 1:
            errh = (err / epst) * epsh * h0
            errd = 1 + 2 * errh
        else:
            errd = h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))
        h *= 0.5;
        m *= 2;
        if not ((errd > errh) & (m < mmax)):
            break
    i *= h
    if errd > errh:
        err = -errd * m
    else:
        err = errh * epsh * m / (2 * efs)
    return i, err

def intdeo(f, a, omega, eps):
    """
    intdeo 
        I = integral of f(x) over (a,infinity),
            f(x) has oscillatory factor :
            f(x) = g(x) * sin(omega * x + theta) as x -> infinity.
        f         : integrand f(x)
        a         : lower limit of integration (double)
        omega     : frequency of oscillation (double)
        eps       : relative error requested (double)
        i         : approximation to the integral
        err       : estimate of the absolute error
        function
            f(x) needs to be analytic over (a,infinity).
        relative error
            eps is relative error requested excluding
            cancellation of significant digits.
            i.e. eps means : (absolute error) /
                             (integral_a^R |f(x)| dx).
            eps does not mean : (absolute error) / I.
        error message
            err >= 0 : normal termination.
            err < 0  : abnormal termination (m >= mmax).
                       i.e. convergent error is detected :
                           1. f(x) or (d/dx)^n f(x) has
                              discontinuous points or sharp
                              peaks over (a,infinity).
                              you must divide the interval
                              (a,infinity) at this points.
                           2. relative error of f(x) is
                              greater than eps.
    """

#     ---- adjustable parameter ----
    mmax = 256; lmax = 5
    efs = 0.1; enoff = 0.40; pqoff = 2.9; ppoff = -0.72
#    /* ------------------------------ */
#    int n, m, l, k;
#    double pi4, epsln, epsh, frq4, per2, pp, pq, ehp, ehm, ir, h, iback,
#        irback, t, ep, em, tk, xw, wg, xa, fp, fm, errh, tn, errd;

    pi4 = math.pi/4
    epsln = 1 - math.log(efs * eps)
    epsh = math.sqrt(efs * eps)
    n = int(enoff * epsln)
    frq4 = abs(omega) / (2 * pi4)
    per2 = math.pi / abs(omega)
    pq = pqoff / epsln
    pp = ppoff - math.log(pq * pq * frq4)
    ehp = math.exp(2 * pq)
    ehm = 1 / ehp
    xw = math.exp(pp - 2 * pi4)
    i = f(a + math.sqrt(xw * (per2 * 0.5)))
    ir = i * xw
    i *= per2 * 0.5
    err = abs(i)
    h = 2
    m = 1
    while 1:
        iback = i
        irback = ir
        t = h * 0.5
        while 1:
            em = math.exp(2 * pq * t)
            ep = pi4 * em
            em = pi4 / em
            tk = t
            while 1:
                xw = math.exp(pp - ep - em)
                wg = math.sqrt(frq4 * xw + tk * tk)
                xa = xw / (tk + wg)
                wg = (pq * xw * (ep - em) + xa) / wg
                fm = f(a + xa)
                fp = f(a + xa + per2 * tk)
                ir += (fp + fm) * xw
                fm *= wg
                fp *= per2 - wg
                i += fp + fm
                if m == 1:
                    err += abs(fp) + abs(fm)
                ep *= ehp
                em *= ehm
                tk += 1
                if (ep >= epsln):
                    break
            if (m == 1):
                errh = err * epsh
                err *= eps
            tn = tk
            while abs(fm) > err:
                xw = math.exp(pp - ep - em)
                xa = xw / tk * 0.5
                wg = xa * (1 / tk + 2 * pq * (ep - em))
                fm = f(a + xa)
                ir += fm * xw
                fm *= wg
                i += fm
                ep *= ehp
                em *= ehm
                tk += 1          
            fm = f(a + per2 * tn)
            em = per2 * fm
            i += em
            if (abs(fp) > err) | (abs(em) > err):
                l = 0
                while 1:
                    l+=1
                    tn += n
                    em = fm
                    fm = f(a + per2 * tn)
                    xa = fm
                    ep = fm
                    em += fm
                    xw = 1
                    wg = 1
                    for k in range(1,n):
                        xw = xw * (n + 1 - k) / k
                        wg += xw
                        fp = f(a + per2 * (tn - k))
                        xa += fp
                        ep += fp * wg
                        em += fp * xw
                    wg = per2 * n / (wg * n + xw)
                    em = wg * abs(em)
                    if (em <= err) | (l >= lmax):
                        break
                    i += per2 * xa
                i += wg * ep
                if (em > err):
                    err = em
            t += h
            if t >= 1:
                break
        if m == 1:
            errd = 1 + 2 * errh
        else:
            errd = h * (abs(i - 2 * iback) + pq * abs(ir - 2 * irback))
        h *= 0.5
        m *= 2
        if not ((errd > errh) & (m < mmax)): 
            break
    i *= h
    if errd > errh:
        err = -errd
    else:
        err *= m * 0.5
    return   i, err


if __name__ == "__main__":    
# intde test 1   nfunc = 0
    def intde_test1(x):
        return 1/math.sqrt(x)
    i,err = intde(intde_test1,0.,1.,1e-15)
    print("result1 = ",i,"error1 = ",err)
#intde test 2    
    def intde_test2(x):
        return math.sqrt(4 - x * x)   
    i,err = intde(intde_test2,0.,2.,1e-15)
    print("result2 = ",i,"error2 = ",err)
#intdei test 1   
    def intdei_test1(x):
        return 1. / (1 + x * x)   
    i,err = intdei(intdei_test1,0.,1e-15)
    print("result3 = ",i,"error3 = ",err)
#intdei test 2
    def intdei_test2(x):
        return math.exp(-x) / math.sqrt(x) 
    i,err = intdei(intdei_test2,0.,1e-15)
    print("result4 = ",i,"error4 = ",err)
#intdeo test 1
    def intdeo_test1(x):
        return math.sin(x)/x 
    i,err = intdeo(intdeo_test1,0.,1.,1e-15)
    print("result5 = ",i,"error5 = ",err)
#intdeo test2
    def intdeo_test2(x):
        return math.cos(x) / math.sqrt(x) 
    i,err = intdeo(intdeo_test2,0.,1.,1e-15)
    print("result6 = ",i,"error6= ",err)
   
