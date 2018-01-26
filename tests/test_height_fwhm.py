#!/usr/bin/env python
from __future__ import (print_function)

from lmfit import models
import inspect
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import sys


# Turn off plotting if run by nosetests.
WITHPLOT = True
for arg in sys.argv:
    if 'nose' in arg or 'pytest' in arg:
        WITHPLOT = False

if WITHPLOT:
    try:
        import matplotlib
        import pylab
    except ImportError:
        WITHPLOT = False


def check_height_fwhm(x, y, model, with_plot=True, report=False):
    """Check height and fwhm parameters"""
    with_plot = with_plot and WITHPLOT
    pars = model.guess(y, x=x)
    out = model.fit(y, pars, x=x)
    if report:
        print(out.fit_report())
    if with_plot:
        fig = pylab.figure()
        out.plot(fig=fig)
        pylab.show()

    # account for functions whose centers are not mu
    mu = out.params['center'].value
    if model is models.LognormalModel():
        cen = np.exp(mu - out.params['sigma']**2)
    else:
        cen = mu
    # get arguments for lineshape
    args = {key: out.best_values[key] for key in
            inspect.getargspec(model.func)[0] if key is not 'x'}
    # output format for assertion errors
    fmt = ("Program calculated values and real values do not match!\n"
           "{:^20s}{:^20s}{:^20s}{:^20s}\n"
           "{:^20s}{:^20f}{:^20f}{:^20f}")

    if 'height' in out.params:
        height_pro = out.params['height'].value
        height_act = model.func(cen, **args)
        diff = height_act - height_pro

        assert abs(diff) < 0.001, fmt.format(model._name, 'Actual',
                'program', 'Diffrence', 'Height', height_act, height_pro, diff)

        if 'fwhm' in out.params:
            fwhm_pro = out.params['fwhm'].value
            func = lambda x:  model.func(x, **args) - 0.5*height_act
            ret = fsolve(func, [cen - fwhm_pro/4, cen + fwhm_pro/2])
            # print(ret)
            fwhm_act = ret[1] - ret[0]
            diff = fwhm_act - fwhm_pro

            assert abs(diff) < 0.5, fmt.format(model._name, 'Actual',
                    'program', 'Diffrence', 'FWHM', fwhm_act, fwhm_pro, diff)

    print(model._name, 'OK')

def test_peak_like():
    # mu = 0
    # variance = 1.0
    # sigma = np.sqrt(variance)
    # x = np.linspace(mu - 20*sigma, mu + 20*sigma, 100.0)
    # y = norm.pdf(x, mu, 1)
    data = np.loadtxt('../examples/test_peak.dat')
    x = data[:, 0]
    y = data[:, 1]
    check_height_fwhm(x, y, models.VoigtModel())
    check_height_fwhm(x, y, models.PseudoVoigtModel())
    check_height_fwhm(x, y, models.Pearson7Model())
    check_height_fwhm(x, y, models.MoffatModel())
    check_height_fwhm(x, y, models.StudentsTModel())
    check_height_fwhm(x, y, models.BreitWignerModel())
    check_height_fwhm(x, y, models.DampedOscillatorModel())
    check_height_fwhm(x, y, models.DampedHarmonicOscillatorModel())
    check_height_fwhm(x, y, models.ExponentialGaussianModel())
    check_height_fwhm(x, y, models.SkewedGaussianModel())
    check_height_fwhm(x, y, models.DonaichModel())
    x=x-9 # Lognormal will only fit peaks with centers < 1
    check_height_fwhm(x, y, models.LognormalModel())

if __name__ == '__main__':
    test_peak_like()
