Wrapper to Python_Option_Pricing for Backtrader
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

File Contains: Python wrapper code for Davis Edward's code for valuation of American options.

 This document demonstrates a Python implementation of some option models described in books written by Davis
 Edwards: "Energy Trading and Investing", "Risk Management in Trading", "Energy Investing Demystified".

 for backward compatability with Python 2.7



A wrapper to price financial options using closed-form solutions written in Python. MIT License.
GBS functions written by Davis Edwards' -- https://github.com/dedwards25/Python_Option_Pricing

The GBS import includes:

European Options: Black-Scholes, Black76, Merton, Garman-Kohlhagan;
Spread Options: Kirk's Approximation, Heat Rate Options;
American Options: Bjerksund-Stensland
Implied Volatility
Asian Options

The wrapper includes only American options, but can be expanded trivially.
