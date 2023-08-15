Code to merge, plot, and analyse key press and sleep data.

## Getting started

I have included an environment.yml file that can be used to create a conda environment. To do so, run:

```console
conda env create -f environment.yml
```

I constructed this file on MacOS, and it gave some problems when running it on the Linux high-performance cluster. I've fixed the problems by simply commenting out some dependencies, but I can't guarantee a completely smooth experience on your own machine.

Alternatively, you can try installing all Python packages manually with pip.

## Fusing key press and sleep data

This happens in the jupyter notebook (clear3_sleep.ipynb). This notebook also generates a set of images per subject that give an indication how well sleep is predicted. I hope the notebook explains itself. If not, let me know.

## Regression

I've set up some quick and dirty mixed-effects models in R. Take a look at the Rmarkdown file regress_sleep.Rmd if you'd like to run these models yourself, or just view regress_sleep.nb.html in your browser (not in Github) to get a quick overview of the results.