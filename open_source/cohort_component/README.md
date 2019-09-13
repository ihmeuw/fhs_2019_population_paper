FHS Cohort Component (CCMP) Model
=========================


driver.py
---------
The main code that accepts all relevant inputs to CCMP (past population, past/future asfr, past/future migration, past/future srb, past/future life table) and evolves the population forward in weekly steps, for a given location.


single.py
---------
Sub-routine for driver.py that evolves population forward one draw at a time.


leslie.py
---------
Contains the construction and arithmetics associated with the Leslie Matrix.


utility methods
---------------
exc.py, interpolate.py, transform.py, tracing.py.
