# SAM-POUNDERS
This is a mostly static repository containing an implementation of SAM-POUNDERS to accompany the paper "Importance Sampling in Expensive Finite-Sum Optimization via Contextual Bandit Methods"

The code in this repository has two primary dependencies:
1) You must clone the [IBCDFO](https://github.com/POptUS/IBCDFO) repository and follow all instructions there to collect its dependencies, in particular MINQ. Ensure that the branch is set to `main`.
2) You must similarly clone the [BenDFO](https://github.com/POptUS/BenDFO) repository and follow all instructions.

Once these repositories are findable on your path, you can run the experiments in Sections 5.2.1 and 5.2.2 by navigating to the `tests` directory, and running `bendfo_test.m`. Be sure to read the headers in that file, as you need to specify in Lines 16 and 20 which experiment to run via the `which_test` string variable, and you should specify however many seeds you want to run through the `num_seeds` variable.  

If you are interested in reproducing Figure 1, you can use `strawman_compare.m`. 

The experiments in Section 5.2.3 are notably much harder to reproduce, not only because it employs a cloud-based LLM, but also because it requires you to supply your own API key to Gemini. You should be able to figure out how to get your own setup working by inspecting `argoGeminiCreateContent.m`. (Argo is a proxy used by Argonne National Laboratory, at the time of writing this paper, to access multiple LLMs). The experiment can then be run by running `oscillator_test.m`. 

The figures in the paper can then be generated from, respectively, `paper_figs.m` and `plot_oscillator_results.m`. 


