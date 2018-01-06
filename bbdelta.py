'''
Author: Benjamin Bradford

MIT License

Copyright (c) 2017 Benjamin Bradford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# File Contains: Python wrapper code for Davis Edward's code for valuation of American options.
#
# This document demonstrates a Python implementation of some option models described in books written by Davis
# Edwards: "Energy Trading and Investing", "Risk Management in Trading", "Energy Investing Demystified".
#
# for backward compatability with Python 2.7
'''


import GBS as GBS
import datetime
import backtrader as bt
import backtrader.indicators as btind


# ----------------------------------------------------------------------------
# Backtrader indicator routines here:
# ----------------------------------------------------------------------------

class bbGreeks(bt.Indicator):


    lines = ('deltaplot', )

    # Riskfreerate for American options should be taken from
    # Overnight Indexed Swaps -- e.g. the Federal Funds rate which
    # is published daily by the Federal Reserve in the US. Overnight
    # rates include EONIA (EUR), SONIA (GBP), CHOIS (CHF), and TONAR (JPY).
    # Do not use LIBOR nor the T-Bill rate.
    params = (('optstring', 'AAPL-20180119-SMART-USD-170-CALL'),
              ('riskfreerate', 0.0142),
              ('dividends', 0.0143), )    # Dividend yield p.a. for AAPL
              #('greek', 1), )           # 0 based index of value, delta, gamma, theta, vega, rho


    def __init__(self):

        #        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=21)

        # Inputs:
        #    option_type = "p" or "c"
        #    fs          = price of underlying
        #    x           = strike
        #    t           = time to expiration
        #    v           = implied volatility
        #    r           = risk free rate
        #    q           = dividend payment
        #    b           = cost of carry

        # Set the option information, strike, expiry, etc.
        cpinf = self.params.optstring.split('-')

        self.expdate = datetime.datetime.strptime(cpinf[1], "%Y%m%d").date()
        # opt type, underlying price, strike, time to expiry, riskfreerate, cost of carry, call or put price:
        self.sopttype = cpinf[5][:1].lower()
        self.doptstrike = float(cpinf[4])

    def next(self):
        #         time to expiration is in years, and must be a float:
        self.time_d = self.expdate - self.data.datetime.date(ago=0)
        print self.time_d, self.expdate, self.data.datetime.date(ago=0)
        timeexp = float(self.time_d.days) / 365

        self.callputprice = float(self.datas[1].close[0])
        self.underlyingprice = float(self.datas[0].close[0])

        #Get implied volatility for american option:
        kwargs = dict(
            option_type=self.sopttype,
            fs=self.underlyingprice,
            x=self.doptstrike,
            t=timeexp,
            r=self.params.riskfreerate,
            q=self.params.dividends,
            cp=self.callputprice, )
        imvol = GBS.amer_implied_vol(**kwargs)

        print "v is ", imvol
        print "fs is ", self.underlyingprice
        print "x is ", self.doptstrike
        print "t is ", timeexp
        print "r is ", self.params.riskfreerate
        print "q is ", self.params.dividends
        print "cp is ", self.callputprice

        kwargs = dict(
            option_type=self.sopttype,
            fs=self.underlyingprice,
            x=self.doptstrike,
            t=timeexp,
            r=self.params.riskfreerate,
            q=self.params.dividends,
            v = imvol, )
        print GBS.american(**kwargs)

        self.lines.deltaplot[0] = GBS.american(**kwargs)[self.greek]


class _delta(bbGreeks):
    def __init__(self):
        self.greek = 1
        bbGreeks.__init__(self)

class _gamma(bbGreeks):
    def __init__(self):
        self.greek = 2
        bbGreeks.__init__(self)

class _theta(bbGreeks):
    def __init__(self):
        self.greek = 3
        bbGreeks.__init__(self)

class _vega(bbGreeks):
    def __init__(self):
        self.greek = 4
        bbGreeks.__init__(self)

class _rho(bbGreeks):
    def __init__(self):
        self.greek = 5
        bbGreeks.__init__(self)