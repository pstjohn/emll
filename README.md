# Linlog modeling code

Some very rough, in-progress work on using linlog kinetics (Wu et al., *Eur. J. Biochem.* **271**, 3348–3359 (2004)) with cobrapy models.

Test models are in `test_models/`. I've been using `pytest` to make sure that any code changes don't break the basic ideas of the linlog model. It essentially runs the different test models and makes sure the steady state solutions calculated via different methods match up where they should.

`linlog_model.py` contains the main code for doing most of the calculations. I also wrote some code to do the same calculations in theano and [casadi](http://casadi.org). (you should be able to `conda install -c conda-forge casadi`, otherwise yell at the [feedstock maintainer](http://github.com/pstjohn))

I'm not including any of the `pymc3` inference code: that's in a really bad state at this point. That's first on my list of things to work on in August though. 
