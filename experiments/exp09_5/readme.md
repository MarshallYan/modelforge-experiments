AIMNet2-QM9

Here we are trying to find the better interaction module configurations just as exp09_4. 
Based on the results from exp09_4, we'd try more configurations for 3 v.s. 5 layers (5 being the better one).

We have seen that decrease the dimensions gradiently may be a good idea, but sometimes that is not necessary.
We will try a bunch of dimension decreases v.s. hold the dimensions relatively higher.

One other factor changed here is that the QM9 energy values are shifted to be maximum at 0 (all negative).
This has been shown to improve test energy MAE in some preliminary test.