import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes, make_1d_axes
from anesthetic.read.getdist import read_paramnames

rootfile = 'Gaussflcdm'
samples = read_chains(rootfile)
params,names = read_paramnames(rootfile)


fig, axes = make_2d_axes(params, upper = False)


samples.plot_2d(axes,c = '#CC3311', #Plotting with recommended resolution settings
                      kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                      diagonal_kwargs={'nplot_1d': 1000},
                      lower_kwargs={'nplot_2d': 100**2},
                      ncompress='entropy')
                      
for j in params: #Adding Lines on the median of each 1d posterior
    median = samples[j].quantile(0.5)
    axes.axlines({j: median}, ls=':', c='black', lw = 0.7, lower=False)   
    
    
fig.savefig(rootfile+'.pdf', format="pdf", bbox_inches = 'tight')
